import string

"""
Script loads in the outer dag file and fills in the number of retries etc.

Launches the optimisation!
"""

NUM_ITERATIONS  = 10 # the number of times to optimise a single variable pair

# create the dag file wth max retries loaded in
with open("dag_templates/main.dag", "r") as infile:
    raw_text = string.Template(infile.read())
out_text = raw_text.substitute(NUM_ITERS = NUM_ITERATIONS)
with open("main.dag", "w") as outfile:
    outfile.write(out_text)

