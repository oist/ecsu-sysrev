src_filename = 'data/SR_DataSet/SR_Duplicates checked.ris'
dst_filename = 'data/SR_DataSet/SR_Duplicates checked numbered.ris'
title_prefix = 'TI  - '
def add_numbers():
    counter = 1
    with open(src_filename) as f_in, open(dst_filename, 'w') as f_out:
        for line in f_in:
            if line.startswith(title_prefix):
                new_line = title_prefix + "{}: ".format(counter) + line[len(title_prefix):]
                f_out.write(new_line)
                counter += 1
            else:
                f_out.write(line)


if __name__ == '__main__':
    add_numbers()
    