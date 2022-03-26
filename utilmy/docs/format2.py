# -*- coding: utf-8 -*-
MNAME = "utilmy.template"
HELP = """ utils for """


import re
from pprint import pprint



def get_file(file_path):
    """function _get_all_line
    Args:
        file_path:   
    Returns:
        
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        all_lines = (f.readlines())
    return all_lines
    

def extrac_block(lines):
    """ Split the code source into Code Blocks:
        header
        import
        variable
        logger
        test
        core

        footer

    """
    dd = {}
    # lines = txt.split("/n")       
    lineblock = []

    flag_test= False

    ## BLOCK HEAD
    for ii,line in enumerate(lines) :
        # print(ii,line)
        # end of block header
        if (re.match(r"import\s+\w+", line) or \
            ((re.match(r"def\s+\w+", line) or re.match(r"class\s+\w+", line) or 'from utilmy import log' in line)) or \
                re.match(r'if __name__', line)) and ii < 20 :
            dd['header'] = lineblock
            dd['header_start_line'] = 0
            dd['header_end_line'] = ii - 1
            lineblock = []
            break
        else:
            lineblock.append(line)
            if ii >= 20:
                dd['header'] = []
                dd['header_start_line'] = 0
                dd['header_end_line'] = 0
                break
    # pprint(dd)
    lineblock = []
    
    # Block import
    dd['import_start_line'] = dd['header_end_line'] + 1 if dd['header_end_line'] else 0
    # print(dd['import_start_line'])
    for ii,line in enumerate(lines) :
        if ii >= dd['import_start_line']:
            # if ('def ' in line or 'class ' in line  or 'from utilmy import log' in line ) and ii < 50 and not 'import' in dd:
            if (re.match(r"def\s+\w+", line) or re.match(r"class\s+\w+", line) or 'from utilmy import log' in line or re.match(r'if __name__', line)) and ii < 50:
                dd['import'] = lineblock
                dd['import_end_line'] = ii - 1
                lineblock = []
                break
            else:
                # print(line)
                lineblock.append(line)
                if ii >= 50:
                    dd['import'] = []
                    dd['header_end_line'] = dd['header_start_line']
                    break
    # pprint(dd)
    lineblock = []

    ### Block Logger
    dd['logger_start_line'] = dd['import_end_line'] + 1 if dd['import_end_line'] else 0
    for ii,line in enumerate(lines) :
        if ii >= dd['logger_start_line']:
            if (re.match(r"def\s+\w+", line)) and ii < 50:
                if not('def help' in line or 'def log' in line):
                    dd['logger'] = lineblock
                    dd['logger_end_line'] = ii - 1
                    lineblock = []
                    break
                else:
                    lineblock.append(line)
            else:
                # print(line)
                lineblock.append(line)
                if ii >= 50:
                    dd['logger'] = []
                    dd['logger_end_line'] = dd['logger_start_line']
                    break
    # pprint(dd)
    lineblock = []

    ### Block Test
    dd['test_start_line'] = dd['logger_end_line'] + 1 if dd['logger_end_line'] else 0
    for ii,line in enumerate(lines) :
        if ii >= dd['test_start_line']:
            # new function / class / or main
            if (re.match(r"def\s+\w+", line) or \
                 re.match(r"class\s+\w+", line)) or \
                 re.match(r'if __name__', line):
                if not('def test' in line):
                    dd['test'] = lineblock
                    dd['test_end_line'] = ii - 1
                    lineblock = []
                    break
                else:
                    lineblock.append(line)
            else:
                # print(line)
                lineblock.append(line)
                if ii == len(lines)-1:
                    dd['test'] = []
                    dd['test_end_line'] = dd['test_start_line']
                    break
    # pprint(dd)

    lineblock = []

    # Block Core
    dd['core_start_line'] = dd['test_end_line'] + 1 if dd['test_end_line'] else 0
    for ii,line in enumerate(lines) :
        if ii >= dd['core_start_line']:
            # new function / class / or main
            if re.match(r'if __name__', line):
                # print('----------------')
                dd['core'] = lineblock
                dd['core_end_line'] = ii - 1
                lineblock = []
                break
            else:
                # print(line)
                lineblock.append(line)
                if ii == len(lines)-1:
                    dd['core'] = []
                    dd['core_end_line'] = dd['core_start_line']
                    break
    # pprint(dd)

    lineblock = []
    dd['footer_start_line'] = dd['core_end_line'] + 1 if dd['core_end_line'] else 0
    for ii,line in enumerate(lines):
        if ii >= dd['footer_start_line']:
            lineblock.append(line)
        if ii == len(lines) -1:
            dd['footer'] = lineblock
            dd['footer_end_line'] = ii
            break
    pprint(dd)

    return dd


def normalize_header(txt):
   #### not need of regex, code easier to read 
   lines = txt.split("\n")

   lines2 = []
   if '# -*- coding: utf-8 -*-' not  in txt :
       lines2.append('# -*- coding: utf-8 -*- ')

   if 'MNAME' not  in txt :   ### MNAME = "utilmy.docs.format"
       nmane =  ".".join( os.path.abspath(__file__).split("/")[-3:] )
       lines2.append( f'MNAME="{mname}" ')

   if 'HELP' not  in txt :   ### HELP
       lines2.append( f'HELP=" util" ')

   ### Add previous line
   lines2 = lines2 + lines
   return "\n".join( lines2)    



def normalize_import(txt):
    """  merge all import in one line and append others

    """
    lines = txt.split("\n")
    
    import_list = [] ; from_list = []
    for line in lines :
      if "import " in line :
         if "from " in line : from_list.append(line)
         else :               import_list.append(line)

            
    #### Merge all import in one line   ################################################
    llall = []
    for imp in import_list :
        ll    = [ t.strip() for t in  imp.split(",") if 'import' not in t ]
        llall = llall + ll

    lall = sorted( lall )
    lines2 = [[]]
    for mi in lall:
       ss =  ss + mi + ","
       if len(ss) >  90 :
          lines2 = lines2.append( ss[:-1] )  
          ss = "import "

    #### Remaining import 
    for ii, line in enumerate(lines):
      if ii > 100 : break
      if line.startswith("import ") : continue  ### Remove Old import
      lines2.append(line)  

    #### 
    return '\n'.join(lines2)




def normalize_logger(txt):
    return txt

def normalize_test(txt):
    return txt

def normalize_core(txt):
    return txt

def normalize_footer(txt):
    return txt


def read_and_normalize_file(file_path, output_file):
    all_lines = get_file(file_path)
    info = extrac_block(all_lines)

    new_headers = normalize_header(info['header'])
    new_imports = normalize_import(info['import'])
    new_loggers = normalize_logger(info['logger'])
    new_tests =   normalize_test(info['test'])
    new_cores =   normalize_core(info['core'])
    new_footers = normalize_footer(info['footer'])

    # Create new data array then write to new file
    new_all_lines = []
    # new_all_lines.extend(new_headers)
    # new_all_lines.extend(new_imports)
    # new_all_lines.extend(new_loggers)
    # new_all_lines.extend(new_tests)
    # new_all_lines.extend(new_cores)
    # new_all_lines.extend(new_footers)

    with open(output_file, 'w+', encoding='utf-8') as f:
        f.writelines(new_all_lines)


if __name__ == '__main__':
    read_and_normalize_file('test_script/test_script_no_core.py', 'test.py')
    read_and_normalize_file('test_script/test_script_no_logger.py', 'test.py')
    # extrac_block(lines)
