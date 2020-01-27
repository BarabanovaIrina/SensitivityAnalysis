from collections import namedtuple, defaultdict
import math
import os
import pandas as pd


def read_vars_inp(path='.'):
    """There must be a function that must reads a VARS input file which includes VARS configuration.
    Function must return a complex variable: dict or namedtuple.
    """

    filename = 'VARS_inp.txt'
    with open(os.path.join(path,filename), 'r', encoding='latin-1') as file:
        raw_text = file.readlines()

    vars_inp = dict(outFldr=raw_text[4],
                    numStars=int(raw_text[5]),
                    grdSize=float(raw_text[6]),
                    IVARS=[float(i) for i in raw_text[7].split()],
                    mdlFile=raw_text[8],
                    mdlFldr=raw_text[9],
                    starFile=raw_text[10],
                    SmpStrtgy=raw_text[11],
                    rndSeed=int(raw_text[12]),
                    btsrpFlg=int(raw_text[13]),
                    btsrpSize=int(raw_text[14]),
                    btsrpCL=float(raw_text[15]),
                    numGrp=int(raw_text[16]),
                    offlineMode=int(raw_text[17]),
                    offlineStage=int(raw_text[18]),
                    calcFreq=int(raw_text[19]),  # min
                    plotFlg=int(raw_text[20]),
                    mdlOutLength=int(raw_text[21]),
                    txtReportFlg=int(raw_text[22]),
                    )
    return vars_inp


def read_factor_space(path):

    """There must be a function extracting factors from factorSpace.csv.
    factor: namedtuple with name, lowerbound and upperbound of factor
    return: list of namedtupeles of factors
    """

    facr_spc_file = 'factorSpace.csv'
    df = pd.read_csv(os.path.join(path, facr_spc_file), encoding='latin-1')
    lst_of_factors = list()
    factor = namedtuple('Factor', ['name', 'lb', 'ub'])
    for index in range(df.shape[0]):
        fct = factor(df.loc[index, 'Name'], df.loc[index, 'LB'], df.loc[index, 'UB'])
        lst_of_factors.append(fct)

    return lst_of_factors


# двоичный исполняемый файл
def vars(inp):
    pass


# двоичный исполняемый файл
def razavi_gupta(inp, out):
    pass


def vars_execution():
    # TODO: vars_inp type
    vars_inp = read_vars_inp()
    fun_path = '.'  # vars_inp.mdlFldr
    # TODO: discuss input files type
    factors = read_factor_space(fun_path)
    vars_inp['factors'] = factors
    vars_out = None

    if vars_inp['mdlOutLength'] == 1:
        vars_out = vars(vars_inp)
    else:
        if vars_inp['offlineMode'] == 0:
            raise ValueError('Online Mode with multi-output SA is not supported by VARS-TOOL-v2.')

        if vars_inp['offlineStage'] == 1:
            vars_out = vars(vars_inp)
        else:
            print(f'VARS with Multi-Output Models: Current Time Step (of {vars_inp["mdlOutLength"]} Total Time Steps) = ')
            ts = defaultdict(str)
            for t in range(1, vars_inp['mdlOutLength']):
                vars_inp['outTime'] = t
                if t > 1:
                    for i in range(0, int(math.log(t-1, 10))):
                        print('\b')
                    print(t)
                    ts[t] = ''  # vars(vars_inp)
            RG = razavi_gupta(vars_inp, vars_out)


