from parser.parser import *

def clean_text(text):
    try:
        if not text:
            return ""

        opening_braces, closing_braces = MatchingDelimiters("[[", "]]")
        remove_external_files = opening_braces + "File:" + Anything() + closing_braces
        text = remove_external_files.remove_from(text)

        opening_braces, closing_braces = MatchingDelimiters("[[", "]]")
        remove_images = opening_braces + "Image:" + Anything() + closing_braces
        text = remove_images.remove_from(text)

        opening_braces, closing_braces = MatchingDelimiters("[[", "]]")
        remove_categories = opening_braces + "Category:" + Anything() + closing_braces
        text = remove_categories.remove_from(text)

        opening_braces, closing_braces = MatchingDelimiters("[[", "]]")
        integrate_links = opening_braces + ReturnValue(Anything()) + closing_braces
        def fix_link(inner):
            if "|" in inner:
                return inner.split("|")[-1]
            return inner
        text = integrate_links.replace_in(text, fix_link)

        opening_braces, closing_braces = MatchingDelimiters("{{", "}}")
        integrate_angbrs = opening_braces + OneOf(["Angbr", "angbr"]) + ReturnValue(Anything()) + closing_braces
        def fix_angbr(inner):
            if "|" in inner:
                return inner.split("|")[-1]
            return ""
        text = integrate_angbrs.replace_in(text, fix_angbr)

        # opening_braces, closing_braces = MatchingDelimiters("{{", "}}")
        # remove_lang_braces = opening_braces + OneOf(["lang", "Lang"]) + ReturnValue(Anything()) + closing_braces
        # def fix_lang_braces(inner):
        #     if "|" in inner:
        #         for part in inner.split("|")[::-1]:
        #             if not ("=" in part or "lang" in part.lower()):
        #                 return part
        #     return ""
        # print(remove_lang_braces.replace_in("{{test}} The first political philosopher to call himself an ''anarchist'' ({{Lang-fr|anarchiste}})", fix_lang_braces))
        # text = remove_lang_braces.replace_in(text, fix_lang_braces)

        opening_braces, closing_braces = MatchingDelimiters("{{", "}}")
        remove_templates = opening_braces + Anything() + closing_braces
        text = remove_templates.remove_from(text)

        opening_braces, closing_braces = MatchingDelimiters("{|", "|}")
        remove_tables = opening_braces + Anything() + closing_braces
        text = remove_tables.remove_from(text)

        for opening, closing in (("<!--", "-->"), ("<ref", "</ref>")):
            opening_tags, closing_tags = MatchingDelimiters(opening, closing)
            remove_html = opening_tags + Anything() + closing_tags
            text = remove_html.remove_from(text)

        opening_braces, closing_braces = MatchingDelimiters("<", ">")
        remove_html = opening_braces + Anything() + closing_braces
        text = remove_html.remove_from(text)

        for junk in ("== References ==", "==References==", "== External links ==", "==External links=="):
            text = remove_below(text, junk)

        if not text:
            print("THERE WAS PROBLEM :(")
            return ""

        text = text.replace("()", "")
        text = text.replace("( , )", "")
        text = text.replace("  ", " ")
        text = text.replace(" , ", ", ")
        text = text.replace("&nbsp;", " ")

        text = remove_excessive_newlines(text)

        text = text.strip()

        return text
    except:
        print("THERE WAS PROBLEM :(\n\nText:\n\n")
        print(text)
        print("\n\n\n\n\n")
        return ""


def remove_below(text, string):
    if string in text:
        return text[:text.index(string)]
    return text

def remove_excessive_newlines(text):
    i=0
    while i < len(text):
        if text[i] == "\n":
            j=i
            while j < len(text) and text[j] == "\n":
                j+=1
            if j-i > 2:
                text = text[:i] + "\n\n" + text[j:]
        i+=1
    return text
