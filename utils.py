import colorama

colorama.init(convert=True)

TITLE_SIDE = "=" * 5 + " "
SUBTITLE_SIDE = ">"*4 + " "

def colored(text, fore=None, back=None, style=None):
    if colorama:
        part = []
        if fore:
            part.append(getattr(colorama.Fore, fore.upper(), None))
        if back:
            part.append(getattr(colorama.Back, back.upper(), None))
        if style:
            part.append(getattr(colorama.Style, style.upper(), None))
        part.append(text)
        part = filter(None, part)
        part.append(colorama.Fore.RESET + colorama.Back.RESET + colorama.Style.RESET_ALL)
        return ''.join(part)
    else:
        return text

def get_title(s):
    return TITLE_SIDE + s + " " + TITLE_SIDE

def get_subtitle(s):
    return SUBTITLE_SIDE + s

def print_title(s):
    print get_title(s)

def print_subtitle(s):
    print get_subtitle(s)

def print_result(cond):
    if cond:
        print colored(get_subtitle("PASSED"), 'green')
    else:
        print colored(get_subtitle("FAILED"), 'red')

def print_testname(name):
    print colored(get_title(name), 'cyan')
