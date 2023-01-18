import os 
import csv

# This function returns the file name from a path 
def get_file_name(filepath):
    return os.path.basename(filepath)

#this function reads a file, and it returns the index of the first line which contains 
#'TIME, CH2, CH4'
def find_first_line(filepath):
    with open(filepath, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if 'TIME,CH2,CH4' in line:
                return i

# This function adds a first line : 'TIME, CH2, CH4' at the beginning of files that don't have it (if find_first_line returns None)
# And saves it in the destination folder
def add_first_line(filepath, dest_folder):
    filename = get_file_name(filepath)
    with open(filepath, 'r') as f:
        with open(dest_folder + '\\' + filename, 'w') as fnew:
            fnew.write('TIME (ms),Voltage (V),Current (mA)\n')
            for line in f.readlines():
                fnew.write(line)

# This function copies a file from a certain line (first_line_index)
# and copies it into a new file which doesn't have the d.csv extension
def write_from_first_line(filepath, first_line_index, dest_folder):
    filename = get_file_name(filepath)
    with open(filepath, 'r') as f:
            with open(dest_folder + '\\' + filename, 'w') as newf:
                newf.write('TIME (ms),Voltage (V),Current (mA)\n')
                for line in f.readlines()[first_line_index+1:]:
                    newf.write(line)

#this functions applies the 2 previous functions depending on whether the first_line_index is none or not
# if it isn't, then it points to the line 'TIME, CH2, CH4' (using find_first_line)
def clean_csv_file(filepath, dest_folder):
    first_line_index = find_first_line(filepath)
    if first_line_index is None :
        add_first_line(filepath, dest_folder)
    else:
        write_from_first_line(filepath, first_line_index, dest_folder)
    print('file :', get_file_name(filepath), 'has been cleaned !')

# This function applies the previous functions to all files with the d.csv extension
def clean_all_files(dir, dest_folder):
    for file in os.listdir(dir):
        if file.endswith('.csv'):
            clean_csv_file(dir + '\\' + file, dest_folder)

#This functions inverts the columns but not the names 
#This function does it for only 1 file 
def invert_vol_cur(filepath):
    filename = get_file_name(filepath)
    rowlist = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for each_row in reader:
            if 'TIME (ms)' in each_row:
                rowlist.append(each_row)
            elif len(each_row) > 2:
                new_row = [each_row[0], each_row[2], each_row[1]]
                rowlist.append(new_row)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rowlist:
            writer.writerow(word for word in row)
    print('file :', filename, 'has been inverted !')

def format_file(filepath):
    # Open the input file in read mode
    with open(filepath, 'r') as input_file:
        # Read the contents of the file into a list of rows
        rows = list(csv.reader(input_file))
    filename = get_file_name(filepath)
    # Open the input file in write mode
    with open(os.path.dirname(filepath) + '\\' + 'f' + filename, 'w') as output_file:
        # Create a CSV writer object
        writer = csv.writer(output_file)
        # Loop through the rows of the input file
        for row in rows:
            if not 'TIME (ms)' in row and len(row) > 2:
                # Multiply the values in the first and second columns by a thousand
                row[0] = float(row[0]) * 1000
                row[2] = float(row[2]) * 1000
            # Write the modified row to the output file
            writer.writerow(row)
    print('file :', get_file_name(filepath), 'has been formatted !')
    os.remove(filepath)

def format_all_files(folder):
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            format_file(folder + '\\' + file)

def shift_time(filepath):
    # Open the CSV file and read the data
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    # Find the first row where the voltage is less than -0.5
    for i, row in enumerate(data):
        if float(row['Voltage (V)']) < -0.5:
            break
    # Calculate the time difference
    time_diff = 152.0 - float(row['TIME (ms)'])
    # Add the time difference to every time value
    for row in data:
        row['TIME (ms)'] = str(float(row['TIME (ms)']) + time_diff)
    # Remove all rows where the time is negative
    data = [row for row in data if float(row['TIME (ms)']) >= 0]
    # Write the modified data back to the file
    filename = get_file_name(filepath)
    with open(os.path.dirname(filepath) + '\\' + 'ts' + filename, 'w') as f:
        fieldnames = ['TIME (ms)', 'Voltage (V)', 'Current (mA)']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print('file :', get_file_name(filepath), 'has been time-shifted !')
    os.remove(filepath)

def shift_all(folder):
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            shift_time(folder + '\\' + file)



def main(input_folder, destination_folder):
    #clean all files
    clean_all_files(input_folder, destination_folder)
    #apply the invert_vol_cur function to all files that end in hz.csv or p.csv
    for file in os.listdir(destination_folder):
        if file.endswith('p.csv') or file.endswith('hz.csv') or file.endswith('s.csv'):
            invert_vol_cur(destination_folder + '\\' + file)
    #inversion comes before formatting 
    format_all_files(destination_folder)
    #after formatting, we shift the time
    shift_all(destination_folder)
    print('all files have been cleaned, formatted and time-shifted !')


import os
if __name__ == '__main__':

    '''first we treat the low_MW files'''
    input_folder = r'.\original_data\22-10-14_low MW_original'
    destination_folder = r'.\low_MW'
    main(input_folder, destination_folder)

    '''then we treat the med_MW files'''
    input_folder = r'.\original_data\22-10-15_med MW_original'
    destination_folder = r'.\med_MW'
    main(input_folder, destination_folder)