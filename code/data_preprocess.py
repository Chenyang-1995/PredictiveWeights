#This py file is used to partition data according to day numbers, one file for each day.
#The keyphases of one record are sorted and concatenated by '-'.

import math
import numpy as np
import numpy.random as random
import mmap
import sys
import pandas as pd
import matplotlib.pyplot as plt

#The maximum day number is 123
DAY_MAX = 123


#represent one line in the file
class Entity:
    def __init__(self,read_line):
        key_phase = read_line[3:-3]
        # print(key_phase)
        key_phase.sort()
        # print(key_phase)
        self.key_phase = '-'.join(key_phase)
        self.day = int(read_line[0])
        self.account_id = read_line[1]
        self.rank = int(read_line[2])
        self.avg_bid = float(read_line[-3])
        self.impression = float(read_line[-2])

    def count_impressions(self):
        if self.rank != 1:
            return self.day,0
        return self.day, self.impression

#represent the orginal data file
class Entities:
    def __init__(self):


        self.day_list = []
        self.account_id_list = []
        self.rank_list = []
        self.keyphrase_list = []
        self.avg_bid_list = []
        self.impressions_list = []

    #add one record
    def add_entity(self,read_line):
        self.day_list.append(read_line[0])
        self.account_id_list.append(read_line[1])
        self.rank_list.append(read_line[2])
        self.impressions_list.append(read_line[-2])
        self.avg_bid_list.append(read_line[-3])
        key_phase = read_line[3:-3]
        #print(key_phase)
        key_phase.sort()
        #print(key_phase)
        key_phase = '-'.join(key_phase)
        self.keyphrase_list.append(key_phase)




    def re_init(self):
        self.day_list = []
        self.account_id_list = []
        self.rank_list = []
        self.keyphrase_list = []
        self.avg_bid_list = []
        self.impressions_list = []

    def write_file(self,filename):
        dataframe = pd.DataFrame({'Day': self.day_list,
                                  'Account_id': self.account_id_list,
                                  'Rank':self.rank_list,
                                  'Keyphrase':self.keyphrase_list,
                                  'Avg_bid': self.avg_bid_list,
                                  'Impressions':self.impressions_list
                                  })

        dataframe.to_csv(filename, index=False, sep=',')


# Returns a list of the newline locations of the input file
def generate_newline_locations(filename):
    # Open the file as a mmap (bytes) to do efficient things counting
    file = open(filename, "r+")
    file_map = mmap.mmap(file.fileno(), 0)

    # iterate through the file for two reasons:
    # 1) to get how many things are actually in the file
    # 2) to record where the newline characters are in the file into a newline_locs file
    newline_locs = [0]
    print("start readline")
    while file_map.readline():
        newline_locs.append(file_map.tell())
    print("finish readline")
    del newline_locs[-1]
    file_map.close()
    return newline_locs


# load lines from a file randomly, input requires set of newline_locs and filename


def load_random_lines(filename, newline_locs, lines):
    # First, figure out the format of the file, if it is a csv it must be split at commas, etc etc.

    try:
        name, extension = filename.split(".")
        if extension == "csv":
            delim = b','
        else:
            delim = None
    except:
        delim = None

    # Open the file as a mmap (bytes) to do efficient processing
    file = open(filename, "r+")
    file_map = mmap.mmap(file.fileno(), 0)

    # generate a random permutation of line indices to grab
    # print(len(newline_locs), " ", lines)
    samples = random.choice(len(newline_locs), lines, replace=False)

    # definies a temporary function which, when given the line number to get, seeks for the correct byte to read the next line and convert to a nparray
    # instead returns the plaintext
    def get_nparray_for_line(linenum):
        file_map.seek(newline_locs[linenum])
        line = file_map.readline()
        return str(line)
        # return np.array([float(y) for y in line.split(delim)])

    # list comprehension to generate the result to return
    results = [get_nparray_for_line(sample) for sample in samples]
    return results

#partition data based on day
def data_day_partition(in_filename,out_filename,selected_day = 0):
    file = open(in_filename, "r+")
    file_map = mmap.mmap(file.fileno(), 0)

    out_file = "{0}_day_{1}.csv".format(out_filename,selected_day)

    line_num = 0
    line = file_map.readline()


    day_list = []
    account_id_list = []
    rank_list = []
    keyphrase_list = []

    data = Entities()
    while line:
        line = str(line, encoding="utf-8")
        line = line.strip('\n')
        # print(line)
        line_list = line.split()
        # print(line_list)
        day = int(line_list[0])

        if day == selected_day:
            data.add_entity(line_list)
            #break

        line_num += 1
        if line_num % 10000000 == 0:
            print("line {}".format(line_num))
        line = file_map.readline()

    data.write_file(out_file)
    data.re_init()

    file_map.close()

#compute the maximum day number
def max_day(filename):
    file = open(filename, "r+")
    file_map = mmap.mmap(file.fileno(), 0)



    day_max = 0
    line_num = 0

    line = file_map.readline()
    while line:
        line = str(line,encoding = "utf-8")
        line = line.strip('\n')
        #print(line)
        line_list = line.split()
        #print(line_list)
        day = int(line_list[0])
        if day > day_max:
            day_max = day
        line_num += 1
        if line_num % 1000000 == 0:
            print("line {}".format(line_num))
        line = file_map.readline()

    print("total_line = {}".format(line_num))
    print("day_max = {}".format(day_max))
    file_map.close()
    return day_max

#obtain the impression number distribution over different days
def impression_num_distribution(in_filename, out_filename='impression_num_distribution.jpg'):
    file = open(in_filename, "r+")
    file_map = mmap.mmap(file.fileno(), 0)

    num_impressions = np.zeros(DAY_MAX,dtype=np.float)

    line_num = 0
    line = file_map.readline()
    while line:
        line = str(line, encoding="utf-8")
        line = line.strip('\n')
        # print(line)
        line_list = line.split()
        # print(line_list)

        curr_line = Entity(line_list)
        day, impression = curr_line.count_impressions()
        num_impressions[day-1] += impression

        line_num += 1
        if line_num % 10000000 == 0:
            print("line {}".format(line_num))
        line = file_map.readline()

    df = pd.DataFrame(num_impressions,
                      index=[i for i in range(1,DAY_MAX+1)],
                      columns=['num_impressions'])

    df.plot()
    plt.savefig(out_filename)

    file_map.close()




def read_data(csv_fname):
    df = pd.read_csv(csv_fname)
    return df

if __name__ == "__main__":
    in_fname = ""

    out_fname = ''



    print("start now!")




    data_day_partition(in_fname,out_fname,selected_day=1)




    print("Done!")