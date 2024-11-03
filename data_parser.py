from idlelib.autocomplete import ATTRS

from lxml import etree
import io, json, os, random, multiprocessing

from sympy.codegen import Attribute

from text_cleaner import clean_text
from tokenizer import encode, decode
import tqdm

DISCARD_REDIRECTS = True
FRACTION_TO_KEEP = 1.0

TRAIN = 0
VALIDATION = 1
TEST = 2

def download_dump_status():
    import requests
    r = requests.get('PUT URL HERE')
    with open('dumpstatus.json', 'wb') as f:
        f.write(r.content)

def get_dump_urls():
    with open('dumpstatus.json') as f:
        data = json.load(f)
    file_names = data['jobs']['articlesmultistreamdump']['files'].keys()
    ret = []
    for file_name in file_names:
        if not "-index" in file_name:
            ret.append("https://dumps.wikimedia.org" + data['jobs']['articlesmultistreamdump']['files'][file_name]['url'])
    return ret

def download_files(urls, num_files=-1):
    import requests
    import os
    import time
    if not os.path.exists('raw_data'):
        os.makedirs('raw_data')
    for i, url in enumerate(urls):
        if num_files != -1 and i >= num_files:
            break
        print("Downloading file", i)
        r = requests.get(url, stream=True)
        with open('raw_data/' + url.split('/')[-1], 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        time.sleep(1)

def load_without_namespace(file):
    parser = etree.XMLParser(remove_blank_text=True)

    tree = etree.parse(file, parser=parser)
    # https://stackoverflow.com/questions/4255277/lxml-etree-xmlparser-remove-unwanted-namespace
    # http://wiki.tei-c.org/index.php/Remove-Namespaces.xsl
    xslt = '''<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="xml" indent="no"/>

    <xsl:template match="/|comment()|processing-instruction()">
        <xsl:copy>
          <xsl:apply-templates/>
        </xsl:copy>
    </xsl:template>

    <xsl:template match="*">
        <xsl:element name="{local-name()}">
          <xsl:apply-templates select="@*|node()"/>
        </xsl:element>
    </xsl:template>

    <xsl:template match="@*">
        <xsl:attribute name="{local-name()}">
          <xsl:value-of select="."/>
        </xsl:attribute>
    </xsl:template>
    </xsl:stylesheet>
    '''

    xslt_doc = etree.parse(io.BytesIO(str.encode(xslt)))
    transform = etree.XSLT(xslt_doc)
    tree = transform(tree)
    root = tree.getroot()
    return tree, root

def unzip_files():
    import bz2
    files = os.listdir('raw_data')
    for file in files:
        if file.endswith('.bz2'):
            print("Unzipping", file)
            with open('./raw_data/' + file, 'rb') as f:
                data = f.read()
            with open('./raw_data/' + file[:-4], 'wb') as f:
                f.write(bz2.decompress(data))
            os.remove('./raw_data/' + file)

def process_file(f):
    print("Processing", f)
    import tqdm
    from text_cleaner import clean_text

    tree, root = load_without_namespace('./raw_data/' + f)

    SHOW_PROGRESS = True
    def generator():
        from tokenizer import encode
        if SHOW_PROGRESS:
            bar = tqdm.tqdm(total=len(root))
        for page in root.iter(tag='page'):
            if SHOW_PROGRESS:
                bar.update(1)
            title = page.find('title').text
            content = page.find('revision').find('text').text

            if DISCARD_REDIRECTS and content.lower().startswith("#redirect"):
                continue

            if random.random() > FRACTION_TO_KEEP:
                continue

            content = clean_text(content)

            if content:
                yield ','.join(map(str, encode(title))), ','.join(map(str, encode(content)))

    stream_to_xml_file(generator(), './cleaned_data/' + ".".join(f.split(".")[:-1]) + ".xml")

class ThreadingFileProcessor:
    def __init__(self, source_file, num_threads):
        self.manager = multiprocessing.Manager()
        self.write_queue = self.manager.Queue()
        self.num_processes = num_threads
        self.source_file = source_file
        self.target_file = './cleaned_data/' + ".".join(source_file.split(".")[:-1]) + ".xml"

    def process(self):
        # tree, root = load_without_namespace('./raw_data/' + self.source_file)

        # new processes
        writer_process = multiprocessing.Process(target=write_to_file, args=(self.write_queue, self.target_file))
        writer_process.start()

        # Create and start processing worker processes
        processes = []
        for proc_index in range(self.num_processes):
            p = multiprocessing.Process(target=process_text, args=(proc_index, self.num_processes, self.write_queue, self.source_file))
            p.start()
            processes.append(p)

        # Wait for all worker processes to finish
        for p in processes:
            p.join()

        # Signal the writer process to stop by putting a sentinel (None) in the queue
        self.write_queue.put(None)
        writer_process.join()

def process_text(proc_index, total_processes, output_queue, source_file):
    # Replace with actual text processing logic
    xml_tree, _ = load_without_namespace('./raw_data/' + source_file)
    pages = xml_tree.xpath(f'/mediawiki/*[position() mod {total_processes} = {proc_index}]')
    for page in pages:
        try:
            title = page.find('title').text
            content = page.find('revision').find('text').text
        except AttributeError:
            continue

        if DISCARD_REDIRECTS and content.lower().startswith("#redirect"):
            continue

        if random.random() > FRACTION_TO_KEEP:
            continue

        content = clean_text(content)

        if content:
            processed_data = {'key': ','.join(map(str, encode(title))), 'value': ','.join(map(str, encode(content)))}
            output_queue.put(processed_data)


def write_to_file(output_queue, output_file_path):
    with open(output_file_path, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<dataset>\n')
        while True:
            data = output_queue.get()
            if data is None:  # Sentinel to signal the end
                break
            # f.write(data + '\n')
            f.write(f'  <datapoint>\n')
            f.write(f'    <key>{data["key"]}</key>\n')
            f.write(f'    <value>{data["value"]}</value>\n')
            f.write(f'    <value_len>{len(data["value"].split(","))}</value_len>\n')
            f.write(f'  </datapoint>\n')
        f.write('</dataset>\n')


def clean_files_threaded(num_threads=10):
    files = os.listdir('raw_data')

    for file in files:
        if "xml" in file.split(".")[-1]:
            t = ThreadingFileProcessor(file, num_threads)
            t.process()


def stream_to_xml_file(generator, file):
    with open(file, 'w', encoding='utf-8') as f:
        # Write the XML declaration and the root start tag
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<dataset>\n')
        for key, value in generator:
            key = escape(key)
            value = escape(value)
            f.write(f'  <datapoint>\n')
            f.write(f'    <key>{key}</key>\n')
            f.write(f'    <value>{value}</value>\n')
            f.write(f'    <value_len>{len(value.split(","))}</value_len>\n')
            f.write(f'  </datapoint>\n')
        f.write('</dataset>\n')

def escape(str_xml: str):
    str_xml = str_xml.replace("&", "&amp;")
    str_xml = str_xml.replace("<", "&lt;")
    str_xml = str_xml.replace(">", "&gt;")
    str_xml = str_xml.replace("\"", "&quot;")
    str_xml = str_xml.replace("'", "&apos;")
    return str_xml

def done_message():
    import subprocess
    subprocess.run(["espeak", '"Process complete"', "-ven-us"])

def error_message():
    import subprocess
    subprocess.run(["espeak", '"Fatal error encountered, terminating process"', "-ven-us"])

def example_data(idx=0):
    tree, root = load_without_namespace("./cleaned_data/" + os.listdir("cleaned_data")[0])
    i = 0
    for datapoint in root.iter(tag="datapoint"):
        if idx == i:
            print(decode(list(map(int, datapoint.find('key').text.split(',')))))
            print(decode(list(map(int, datapoint.find('value').text.split(',')))))
            break
        i += 1

if __name__ == '__main__':
    example_data(10)
    # try:
        # download_files(get_dump_urls(), 1)
        # unzip_files()
        # clean_files_threaded(num_threads=10)
        # done_message()
        # for title, text in generate_data(split=VALIDATION):
        #     print(title, text)
        #     break
    # except Exception as e:
    #     print(e)
        # error_message()
