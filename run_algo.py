import string

"""
Script loads in the outer dag file and fills in the number of retries etc.
"""

full_algo_repeats  = 2 # number of times entire tuning algo runs
pair_block_repeats = 1 # number of times each block is repeated ([t1,t2,a1,a2], [t3, t4, a4,a3], [tr]) 
sample_space_iters = 20 # number of times each parameter tuning loop is repeated within the blocks 
# create the dag file wth max retries loaded in
with open("dag_templates/main.dag", "r") as infile:
    raw_text = string.Template(infile.read())
out_text = raw_text.substitute(NUM_ITERS = full_algo_repeats)
with open("dag_running/main.dag", "w") as outfile:
    outfile.write(out_text)

# create the dag file wth max retries loaded in
with open("dag_templates/algo.dag", "r") as infile:
    raw_text = string.Template(infile.read())
out_text = raw_text.substitute(NUM_ITERS = pair_block_repeats, RISE_REPEATS = 0)
with open("dag_running/algo.dag", "w") as outfile:
    outfile.write(out_text)

# create the dag file wth max retries loaded in
with open("dag_templates/first_pair.dag", "r") as infile:
    raw_text = string.Template(infile.read())
out_text = raw_text.substitute(NUM_ITERS = sample_space_iters)
with open("dag_running/first_pair.dag", "w") as outfile:
    outfile.write(out_text)

# create the dag file wth max retries loaded in
with open("dag_templates/second_pair.dag", "r") as infile:
    raw_text = string.Template(infile.read())
out_text = raw_text.substitute(NUM_ITERS = sample_space_iters)
with open("dag_running/second_pair.dag", "w") as outfile:
    outfile.write(out_text)

# create the dag file wth max retries loaded in
with open("dag_templates/rise_time.dag", "r") as infile:
    raw_text = string.Template(infile.read())
out_text = raw_text.substitute(NUM_ITERS = sample_space_iters)
with open("dag_running/rise_time.dag", "w") as outfile:
    outfile.write(out_text)

# create the dag file wth max retries loaded in
with open("dag_templates/opto.dag", "r") as infile:
    raw_text = string.Template(infile.read())
out_text = raw_text.substitute()
with open("dag_running/opto.dag", "w") as outfile:
    outfile.write(out_text)
