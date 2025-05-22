from colorama import init, Fore, Style
import sys,time,random
import sys
import regex as re
import asyncio


# code from https://gist.github.com/jeffskinnerbox/6663095
colorCodes = {
    'black':     '0;30',    'bright gray':   '0;37',
    'blue':      '0;34',    'white':         '1;37',
    'green':     '0;32',    'bright blue':   '1;34',
    'cyan':      '0;36',    'bright green':  '1;32',
    'red':       '0;31',    'bright cyan':   '1;36',
    'purple':    '0;35',    'bright red':    '1;31',
    'yellow':    '0;33',    'bright purple': '1;35',
    'dark gray': '1;30',    'bright yellow': '1;33',
    'normal':    '0'
}

########## print thought, will not log ##########
#typing_speed: wpm
def slow_type_approximation(t):
    t = '[Approximation agent:]\n' + t + '\n[End Thought Process]'
    for l in t:
        sys.stdout.write("\033[" + colorCodes['bright cyan'] + "m" + l + "\033[0m")
        #sys.stdout.write(l)
        sys.stdout.flush()
        time.sleep(random.random()*10.0/300)
    print('')
    return ''

def slow_type_target(t):        
    t = '[Target agent:]\n' + t + '\n[End Thought Process]'
    for l in t:
        sys.stdout.write("\033[" + colorCodes['bright purple'] + "m" + l + "\033[0m")
        #sys.stdout.write(l)
        sys.stdout.flush()
        time.sleep(random.random()*10.0/300)
    print('')
    return ''

if __name__ == '__main__':
    slow_type_approximation('This is a test')
    slow_type_target('This is a test')