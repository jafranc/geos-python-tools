import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from itertools import chain


def process(fname):
    buffer, time_tag, _ = _split_log_(fname)
    expressions = _init_expressions_()
    collabels = list(expressions.keys())
    collabels = _expand_labels(collabels)
    collabels.insert(0,'time')
    values = pd.DataFrame(columns=collabels)

    for i, bits in enumerate(buffer):
        new_row = {}
        new_row['time'] = time_tag[i + 1]
        for key, exp in expressions.items():
            value = _fetch_masses_(bits, exp)
            if value is not None:
                if isinstance(value,float):
                   new_row[key] = value
                else :
                    for icol in range(value.shape[0]):
                        new_row[key+'_{}'.format(icol)] = value[icol]
        if len(new_row)==len(values.columns):
            values = pd.concat([pd.DataFrame(new_row,columns=values.columns,index=[i]), values.loc[:]]).reset_index(drop=True)

    __plot_dataframe__(values,'time',['mobile_0','immobile_0','dissolved_2','mobile_0+immobile_0+dissolved_2'])

def __plot_dataframe__(values, xkey, ykeys):

    for k in ykeys:
        yfield = 0*values['time']
        if re.findall(r'\+',k):
            subkeys = k.split('+')
            for sk in subkeys:
                yfield += values[sk]
        else:
            yfield = values[k]

        plt.plot(values[xkey].to_numpy(),yfield.to_numpy(),label=k)
    plt.legend()
    plt.show()

#todo  suppport complex operations
# def __operation_on_key__(values, keys):
#
#     for k in keys:
#         if re.match(r'\*',k):
#             __operation_on_key__(values, k.split('*'))
#         elif re.match(r'\/',k):
#             __operation_on_key__(values, k.split('/'))
#         elif re.match(r'\+',k):
#             __operation_on_key__(values,k.split('+'))
#         elif re.match(r'-',k):
#             __operation_on_key__(values,k.split('-'))

def _expand_labels(labels):
    new_labels = []
    for i,label in enumerate(labels):
        if re.match(r'total|immobile|mobile|trapped',label):
            new_labels.extend([label+'_0',label+'_1'])
        elif re.match(r'dissolved',label):
            #todo read nphase and ncp in header bit
            new_labels.extend([label+'_{}'.format(xc) for xc in range(4)])
        else:
            new_labels.append(label)

    return new_labels

def _init_expressions_():
    expression = {}
    universal_float = r'\d+(\.)?(\d+)?(e[+-]\d+)?'
    expression[
        'total'] = r'reservoir[1-9]: Phase mass: \{ (' + universal_float + '), (' + universal_float + ') \} kg'
    expression[
        'trapped'] = r'reservoir[1-9]: Trapped phase mass \(metric 1\): \{ (' + universal_float + '), (' + universal_float + ') \} kg'
    expression[
        'immobile'] = r'reservoir[1-9]: Immobile phase mass \(metric 2\): \{ (' + universal_float + '), (' + universal_float + ') \} kg'
    expression[
        'mobile'] = r'reservoir[1-9]: Mobile phase mass \(metric 2\): \{ (' + universal_float + '), (' + universal_float + ') \} kg'
    expression[
        'dissolved'] = r'reservoir[1-9]: Dissolved component mass: \{ \{ (' + universal_float + '), (' + universal_float + ') \}, \{ (' + universal_float + '), (' + universal_float + ') \} \} kg'
    return expression


def _split_log_(fname):
    buffer = []
    time_tag = {}
    buffer_ = ''
    with open(fname) as f:
        for line in f:
            if re.match(r'^Time', line):
                buffer.append(buffer_)
                time_tag[len(buffer)] = float(re.match(r'^Time: (\d+\.\d+e[+-]\d+)', line).group(1))
                buffer_ = ''
            elif re.match(r'Cleaning up events', line):
                buffer.append(buffer_)
            else:
                buffer_ += line
    header = buffer.pop(0)  # discard pre-simulation values
    return buffer, time_tag, header


def _fetch_masses_(buffer, expression):
    trapped = []
    t = re.findall(expression, buffer)
    if len(t)>0:
        for matched in t:
            trapped.append([float(matched[i]) for i in range(0,len(matched),4)])
    else:
        return None
    return np.sum(np.asarray(trapped), axis=0)


if __name__ == "__main__":
    process(fname='samples/slurm-spe11b-volume.out')
