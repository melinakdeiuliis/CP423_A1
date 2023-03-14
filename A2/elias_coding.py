import math
import argparse

log2 = lambda x: math.log(x, 2)

# Convert x to unary value
def Unary(x):
    return (x-1) * '0' + '1'

# Convert x to binary value
def Binary(x, l = 1):
    s = '{0:0%db}' % l
    return s.format(x)

def Binary_Without_MSB(num):
    binary = "{0:b}".format(int(num))
    binary_without_MSB = binary[1:]
    return binary_without_MSB

# Status: Verified
def gamma_encode(num):
    """
    Encodes a positive integer using Elias Gamma coding
    """
    if (num == 0):
        return '0'
    
    elif (num == 1):
        return '1'

    l = int(log2(num))

    n = 1 + l
    b = num - 2 ** l

    return (Unary(n) + Binary(b, l))

# Status: Verified
def gamma_decode(num):
    """
    Decodes a binary string using Elias Gamma coding
    """

    N = 0
    num = list(num)

    # Count number of leading zeroes in num
    while True:
        if not num[N] == '0':
            if num[N] == '1':
                break
            else:
                return 'ERROR'

        N += 1

    # Reading bits starting at the first 1
    num = num[N:2*N+1]
    print(num)

    # Reverse num list
    num.reverse()

    decimal = 0

    # Binary num to integer
    for i in range(len(num)):
        if num[i] == '1':
            decimal += math.pow(2, i)

    return int(decimal)

# Status: Verified
def delta_encode(num):
    """
    Encode a positive integer using Elias Delta coding
    """

    if (num == 0):
        return "0"
    
    elif (num == 1):
        return "1"

    gamma = gamma_encode(1 + math.floor(log2(num)))

    return gamma + Binary_Without_MSB(num)

# Status: Verified
def delta_decode(num):
    '''
    Decode a binary string using Elias Delta decoding
    '''
    N = 0
    num = list(num)

    # Count number of leading zeroes in num
    while True:
        if len(num) > 1 and num[0] == '1': return 'Error'
        elif not num[N] == '0':
            if num[N] == '1':
                break
            else:
                return 'ERROR'
        

        N += 1

    L = num[2*N+1:]

    # Insert 1 to MSB in L to represent 2^N
    L.insert(0, '1')

    # Reverse list to read binary sequence
    L.reverse()
    
    decode = 0

    # Convert binary sequence to integer
    for i in range(len(L)):
        if L[i] == '1':
            decode += math.pow(2, i)

    return int(decode)

parser = argparse.ArgumentParser()
parser.add_argument('--alg', type = str, choices = ['gamma', 'delta'], required = True, help = 'Choose the algorithm (gamma or delta)')

options = parser.add_mutually_exclusive_group(required = True)
options.add_argument('--decode', type = str, nargs = '+', help = 'Decode a sequence of codes')
options.add_argument('--encode', type = int, nargs = '+', help = 'Encode an array of integers')

args = parser.parse_args()

if args.decode is not None:
    code_values = args.decode
    if args.alg == 'gamma':
        for x in code_values:
            print(gamma_decode(x))
    else:
        for x in code_values:
            print(delta_decode(x))

if args.encode is not None:
    int_values = args.encode
    if args.alg == 'gamma':
        for x in int_values:
            print(gamma_encode(x))
    else:
        for x in int_values:
            print(delta_encode(x))