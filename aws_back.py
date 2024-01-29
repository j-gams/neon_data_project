import os
import sys
aws_pem_loc = '"/Users/Jerry/aws_pems/jega7451.firerx.pem"'

instance_num =int(sys.argv[1])
getdir = sys.argv[2]
instance_list = ["ubuntu@ec2-52-10-17-184.us-west-2.compute.amazonaws.com",
                 "ubuntu@ec2-52-24-100-137.us-west-2.compute.amazonaws.com",
                 "ubuntu@ec2-54-149-167-121.us-west-2.compute.amazonaws.com",
                 "ubuntu@ec2-54-184-127-233.us-west-2.compute.amazonaws.com",
                 "ubuntu@ec2-54-68-14-126.us-west-2.compute.amazonaws.com"]

getfrom = ":/home/ubuntu/models/trained/"
aws_instance = instance_list[instance_num]
local_base = "/Users/jerry/Desktop/work/earthlab/neon_data_project/models_3/trained/"


localmod = getdir

os.system("scp -i " + aws_pem_loc + " -r " + aws_instance + getfrom + getdir + " " + local_base + getdir)