import csv
import pickle

csv_file_name = './data/input.csv'
output_file_name = './data/onehot_data.pkl'

readfile = open(csv_file_name, 'r')
writefile = open(output_file_name, 'wb+')
reader = csv.DictReader(readfile)


sets = {
    'Category': set(),
    'Cause': set(),
    'Focus': set()
}
setkeys = ['Category', 'Cause', 'Focus']
total_lines = 0
for line in reader:
    for key in setkeys:
        sets[key].add(line[key])
    total_lines += 1

for key in setkeys:
    all_items = sets[key]
    dic = {}
    for index, item in enumerate(all_items):
        dic[item] = index
    dic['total'] = len(all_items)
    sets[key] = dic

pickle.dump(sets, writefile)
readfile.close()
writefile.close()