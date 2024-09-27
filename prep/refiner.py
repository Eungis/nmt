import re


def read_regex(fn):
    """
    Args:
        fn (str): Regex file name. It must be separated in two columns with delimiter "\t".
    Func:
        Read regex file
    """
    regexs = []

    f = open(fn, "r")

    for line in f:
        if not line.startswith("#"):
            tokens = line.split("\t")

            if len(tokens) == 1:
                tokens += [" "]

            tokens[0] = tokens[0][:-1] if tokens[0].endswith("\n") else tokens[0]
            tokens[1] = tokens[1][:-1] if tokens[1].endswith("\n") else tokens[1]

            regexs += [(tokens[0], tokens[1])]

    f.close()

    return regexs


def refine(data, column, regex_fn):
    """
    Args:
        data (dataframe): Target data (pandas dataframe)
        column (str): Target column of the data
        regex_fn: Regex file name. It must be separated in two columns with delimiter "\t".
    Returns:
        Read regex file and refine the column of the data.
        Return refined sentences with list.
    """

    tgt = data[column]
    regexs = read_regex(regex_fn)

    lines = []
    for line in tgt:
        if line.strip() != "":
            for regex in regexs:
                line = re.sub(f"{regex[0]}", f"{regex[1]}", line)
                line = line.replace("\n", " ").strip()
            lines += [line]
        else:
            lines += [""]

    return lines
