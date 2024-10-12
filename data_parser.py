from lxml import etree
import io, json, os, random

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


def clean_files():
    import multiprocessing

    files = os.listdir('raw_data')

    xml_files = []

    for file in files:
        if "xml" in file.split(".")[-1]:
            xml_files.append(file)

    with multiprocessing.Pool() as pool:
        pool.map(process_file, xml_files)

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


if __name__ == '__main__':
    # try:
        # download_files(get_dump_urls(), 1)
        # unzip_files()
        clean_files()
        done_message()
        # for title, text in generate_data(split=VALIDATION):
        #     print(title, text)
        #     break
    # except Exception as e:
    #     print(e)
        # error_message()
