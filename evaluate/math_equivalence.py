"""
    This script contains the function is_equivalent that compares two math strings and returns True if they are equivalent.
"""
import traceback
import re

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):

    if not ("\\sqrt" in string or 'sqrt(' in string):
        return string

    string = re.sub(r'sqrt\(([^)]*)\)', r'\\sqrt{\1}', string)

    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    original_string = string

    string = string.lower()

    # strip final . or , or ;
    if string[-1].strip() in [".", ",", ";"]:
        string = string.strip()[:-1]

    # linebreaks
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # replace numbers with their actual values
    string = string.replace("doua", "2")
    string = string.replace("doi", "2")
    string = string.replace("a doua", "2")
    string = string.replace("al doilea", "2")
    string = string.replace("trei", "3")
    string = string.replace("a treia", "3")
    string = string.replace("al treilea", "3")
    string = string.replace("patru", "4")
    string = string.replace("a patra", "4")
    string = string.replace("al patrulea", "4")
    string = string.replace("cinci", "5")
    string = string.replace("sase", "6")
    string = string.replace("sapte", "7")
    string = string.replace("opt", "8")
    string = string.replace("noua", "9")
    string = string.replace("zece", "10")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove math delimiters
    string = string.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace('\\]', "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("%", "")

    # Do something about \operatorname{} and \mathrm{} and \text{} and \textnormal{} and \textbf{} and \textit{} and \text
    string = re.sub(r'\\operatorname{([^}]*)}', r'\1', string)
    string = re.sub(r'\\mathrm{([^}]*)}', r'\1', string)
    string = re.sub(r'\\textnormal{([^}]*)}', r'\1', string)
    string = re.sub(r'\\textbf{([^}]*)}', r'\1', string)
    string = re.sub(r'\\textit{([^}]*)}', r'\1', string)
    string = re.sub(r'\\text{([^}]*)}', r'\1', string)

    # Do something about \{ \}
    string = string.replace("\\{", "{").replace("\\}", "}")

    # Do something about cos, sin, tan, etc.
    string = string.replace("\\cos", "cos")
    string = string.replace("\\sin", "sin")
    string = string.replace("\\tan", "tan")
    string = string.replace("\\cot", "cot")
    string = string.replace("\\sec", "sec")
    string = string.replace("\\csc", "csc")
    string = string.replace("\\arccos", "arccos")
    string = string.replace("\\arcsin", "arcsin")
    string = string.replace("\\arctan", "arctan")
    string = string.replace("\\arccot", "arccot")
    string = string.replace("\\arcsec", "arcsec")
    string = string.replace("\\arccsc", "arccsc")
    string = string.replace("\\cosh", "cosh")
    string = string.replace("\\sinh", "sinh")
    string = string.replace("\\tanh", "tanh")

    # Do something about cos x, cos(x)
    string = re.sub(r'cos\(([^)]*)\)', r'cos\1', string)
    string = re.sub(r'sin\(([^)]*)\)', r'sin\1', string)
    string = re.sub(r'tan\(([^)]*)\)', r'tan\1', string)
    string = re.sub(r'cot\(([^)]*)\)', r'cot\1', string)

    # Do something about \rfloor and \lfloor
    string = string.replace("\\lfloor", "[")
    string = string.replace("\\rfloor", "]")

    # Do something about \pm, \lt, \gt, \leq, \geq
    string = string.replace("\\pm", "±")
    string = string.replace("\\lt", "<")
    string = string.replace("\\gt", ">")
    string = string.replace("\\leq", "≤")
    string = string.replace("\\geq", "≥")

    # Do something about \cdot and \times
    string = string.replace("\\cdot", "*")
    string = string.replace("\\times", "*")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    # might be the case that on the left side we have a function name, e.g. "cos x = " or "sin x = "
    if len(string.split(",")) == (len(string.split("=")) - 1) and len(string.split(",")) > 1 and len(string.split("=")) > 2:
        # 'a=2, b=3, c=4' --> '2,3,4'
        string = ",".join([x.split("=")[-1] for x in string.split(",")])
    elif len(string.split("=")) >= 2:
        string = string.split("=")[-1]

    # get rid of x \in {} or x \in [] or x \in ()
    if len(string.split("\\in")) == 2:
        # check if string[1] is surrounded by {} or [] or ()
        if string.split("\\in")[1][0] in ["{", "[", "("] and string.split("\\in")[1][-1] in ["}", "]", ")"]:
            string = string.split("\\in")[1]

    # if it is a set, order the elements and remove spaces
    if string[0] == '{' and string[-1] == '}':
        string = "{" + ",".join(
            sorted(map(lambda x: x.strip(), string[1:-1].split(",")))
        ) + "}"

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    has_negative_sign = string.startswith("-")
    if has_negative_sign:
        string = string[1:]

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # Notable cases: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875
    known_fractions = {'0.1': '\\frac{1}{10}', '0.2': '\\frac{1}{5}', '0.4': '\\frac{2}{5}', '0.5': '\\frac{1}{2}', '0.6': '\\frac{3}{5}', '0.7': '\\frac{7}{10}', '0.8': '\\frac{4}{5}', '0.9': '\\frac{9}{10}', '0.25': '\\frac{1}{4}', '0.75': '\\frac{3}{4}', '0.125': '\\frac{1}{8}', '0.375': '\\frac{3}{8}', '0.625': '\\frac{5}{8}', '0.875': '\\frac{7}{8}'}
    if string in known_fractions:
        string = known_fractions[string]

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    if has_negative_sign:
        string = "-" + string

    # 25ani --> 25
    if re.sub(r'[a-zăâîșț]*\d+[a-zăâîșț]+' , '', string) == '':
        string = re.sub(r'[a-zăâîșț]+' , '', string).strip()

    # '672, 671 respectiv 669 monede' --> '669,671,672'
    if re.sub(r'\d+[^0-9]+', '', string) == '' and '*' not in string and '^' not in string and '/' not in string and '+' not in string and '-' not in string:
        string = ",".join(sorted(re.findall(r'\d+', string)))

    # 'AAAAASSB și SSSSSALL' --> 'AAAAASSB,SSSSSALL'
    if 'și' in string:
        string = ",".join(sorted(string.split('și')))

    # '6928*d*i**2*v-35616*i**3*v' --> '6928\cdot d\cdot i^{2}\cdot v - 35616\cdot i^{3}\cdot v'
    if '*' in string:
        string = string.replace('**', '^').replace('*', '\cdot ')
        if "^" in string and "^{" not in string:
            string = re.sub(r"\^\d+", lambda x: "^{" + x.group()[1:] + "}", string)

    #remove space before and after '-', '+', '*', '/'
    string = string.replace(' -', '-').replace(' +', '+').replace(' *', '*').replace(' /', '/').replace('- ', '-').replace('+ ', '+').replace('* ', '*').replace('/ ', '/')

    return string

def is_equivalent(str1, str2, verbose = False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True

    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)

        if verbose:
            print(ss1, ss2)

        return ss1 == ss2
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        return str1 == str2

if __name__ == "__main__":
    # # Testing
    s1 = "\\frac{1}{2}"
    s2 = "0.5"
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = '\\(x=-3\\)'
    s2 = '-3'
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = '\\(\\operatorname{det}(\\mathrm{B})=3\\)'
    s2 = '\\mathrm{det}(\\mathrm{B})=3'
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = '\\(\\operatorname{det}(\\mathrm{B})=3\\)'
    s2 = 'det(B)=3'
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = '\\(\\cos x=-\\frac{4}{5}\\)'
    s2 = 'cos(x)=-0.8'
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = 'cos x = -\\frac{4}{5}'
    s2 = '-0.8'
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = '\\(\\lfloor\\sqrt{2}+\\sqrt{3}\\rfloor=3\\)'
    s2 = '3'
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = '\\(\\lfloor\\sqrt{2}+\\sqrt{3}\\rfloor\\)'
    s2 = '[sqrt(2)+sqrt(3)]'
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = '\\(x \\in\\{-1,1,-\\sqrt{2}, \\sqrt{2}\\}\\)'
    s2 = '{1, -1,-sqrt(2),sqrt(2)}'
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = '\\(f(1)=\\mathrm{e}\\)'
    s2 = 'e'
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = '64 .'
    s2 = '64'
    assert is_equivalent(s1, s2), s1 + " " + s2

    # TODO still not working
    # s1 = '\\(x_{1,2}=\\frac{-5 \\pm i \\sqrt{3}}{2}\\)'
    # s2 = '0.5 * (-5 ± isqrt(3))'
    # assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = '\(\\operatorname{Din} \\mathbf{e}) \\Rightarrow f_{n}(-1)=(-1+1)^{2^{n}}-1=-1\)'
    s2 = '-1'
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = 'a=2, b=3, c=4'
    s2 = '2,3,4'
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = 'a=2015, b=2016, c=2017, d=2011.'
    s2 = '2015,2016,2017,2011'
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = '25ani'
    s2 = '25'
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = '49 de triunghiuri ca în enunț'
    s2 = '49'
    assert is_equivalent(s1, s2), s1 + " " + s2

    # s1 = 'x=10, y=20, sau x=20, y=10'
    # s2 = '10,20'
    # assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = '672, 671 respectiv 669 monede'
    s2 = '669,671,672'
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = 'AAAAASSB și SSSSSALL'
    s2 = 'AAAAASSB,SSSSSALL'
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = 'în a treia săptămâna'
    s2 = '3'
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = "doi, trei"
    s2 = "2,3"
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = '-6/17'
    s2 = '-\\frac{6}{17}'
    assert is_equivalent(s1, s2), s1 + " " + s2

    s1 = '6928*d*i**2*v - 35616*i**3*v'
    s2 = '6928\cdot d\cdot i^{2}\cdot v - 35616\cdot i^{3}\cdot v'
    assert is_equivalent(s1, s2), s1 + " " + s2