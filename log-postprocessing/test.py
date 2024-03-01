import re

def process(sin):
    if re.findall(r'\+',sin):
        bits = sin.split('+')
        return process(bits[0]) + ' + '+ process('+'.join(bits[1:]))
    elif re.findall(r'-',sin):
        bits = sin.split('-')
        return process(bits[0]) + ' - '+ process('+'.join(bits[1:]))
    elif re.findall(r'\*',sin):
        bits = sin.split('*')
        return  process(bits[0]) + ' * '+ process('*'.join(bits[1:]))
    elif re.findall(r'/',sin):
        bits = sin.split('/')
        return  process(bits[0]) + ' / '+ process('/'.join(bits[1:]))
    return sin

if __name__ == "__main__":
    test = 'a+b*c/d*f+g'
    print(process(test))

    test = 'a-b*c*d*e+z'
    print(process(test))