import os
import sys
aws_pem_loc = '"/Users/Jerry/aws_pems/jega7451.firerx.pem"'

### TODO -
# (3) - cube - 3, 4 - 4, 3
# (3) - pyr - 1, 2 - 2, 1
# (3) - pac - 5, 6 - 6, 5
# (3) - cascade - 7 - 7

#box1 ssh -i "jega7451.firerx.pem" ubuntu@ec2-52-10-17-184.us-west-2.compute.amazonaws.com
#box2 ssh -i "jega7451.firerx.pem" ubuntu@ec2-52-24-100-137.us-west-2.compute.amazonaws.com
#box3 ssh -i "jega7451.firerx.pem" ubuntu@ec2-54-149-167-121.us-west-2.compute.amazonaws.com
#box4 ssh -i "jega7451.firerx.pem" ubuntu@ec2-54-184-127-233.us-west-2.compute.amazonaws.com
#box5 ssh -i "jega7451.firerx.pem" ubuntu@ec2-54-68-14-126.us-west-2.compute.amazonaws.com
#ssh -i "jega7451.firerx.pem" ubuntu@ec2-54-149-167-121.us-west-2.compute.amazonaws.com
#box6 using box1 --
#box7 using box3 --

instance_num = int(sys.argv[1])
instance_list = ["ubuntu@ec2-52-10-17-184.us-west-2.compute.amazonaws.com",
                 "ubuntu@ec2-52-24-100-137.us-west-2.compute.amazonaws.com",
                 "ubuntu@ec2-54-149-167-121.us-west-2.compute.amazonaws.com",
                 "ubuntu@ec2-54-184-127-233.us-west-2.compute.amazonaws.com",
                 "ubuntu@ec2-54-68-14-126.us-west-2.compute.amazonaws.com"]
destination = ":/home/ubuntu/models"

aws_instance = instance_list[instance_num]

local_base = "/Users/jerry/Desktop/work/earthlab/neon_data_project/"
files_transfer = ["models_3/model_frame.py",
                  "models_3/model_bank.py",
                  "models_3/data_handler.py"]
#files_transfer = ["visualize/viz_functions.py",
#                  "visualize/main_viz.py"]
directories_transfer = []#["models_3"]

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
for file in files_transfer:
    os.system("scp -i " + aws_pem_loc + " " + local_base + file + " " +
              aws_instance + destination)


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