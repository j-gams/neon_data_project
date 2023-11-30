import os

aws_pem_loc = '"/Users/Jerry/aws_pems/jega7451.firerx.pem"'
aws_instance = "ubuntu@ec2-54-149-167-121.us-west-2.compute.amazonaws.com"
destination = ":/home/ubuntu/models"

local_base = "/Users/jerry/Desktop/work/earthlab/neon_data_project/"
files_transfer = [""]
directories_transfer = ["models_3"]

check_hist = False
hist_loc = "aws_progress.txt"

ignore_hidden = True

completed = []
if check_hist:
    try:
        with open(hist_loc, 'r') as histfile:
            completed = histfile.read().split(",")
    except:
        print("no file to begin from")
else:
    os.system("rm " + hist_loc)

fail_limit = 10

#"/Users/jerry/Desktop/work/earthlab/neon_data_project/data/pyramid_sets/box_pyramid"

print("* TRANSFERING LOOSE FILES")
f_list = []
for file in files_transfer:
    pass

print("* TRANSFERING DIRECTORIES")
df_list = []
hist_found = 0
for directory in directories_transfer:
    for (dirpath, dirnames, filenames) in os.walk(directory):
        flen = len(dirpath + "/")
        filenames = [dirpath + "/" + f for f in filenames]
        atelt = 0
        while atelt < len(filenames):
            if filenames[atelt] in completed:
                filenames.pop(atelt)
                hist_found += 1
            elif ignore_hidden and filenames[atelt][flen] == ".":
                filenames.pop(atelt)
            else:
                atelt += 1
        df_list.extend(filenames)
        #print(filenames)
        print(dirpath)
        break

print("found", hist_found, "successfully uploaded files")
# now upload...
file_on = 0
fail_at = 0
print(len(df_list), "files to transfer")
while file_on < len(df_list):
    #print("attempting file", file_on+1, "out of", len(df_list), end=": ")
    result = os.system("scp -i " + aws_pem_loc + " " + local_base + df_list[file_on] + " " +
                       aws_instance + destination)
    if result == 0:
        file_on += 1
        fail_at = 0
        with open(hist_loc, 'a') as histfile:
            histfile.write(df_list[file_on-1]+",")
    else:
        print("failed! trying again.")
        fail_at += 1
        if fail_at > fail_limit:
            fail_at = 0
            file_on += 1