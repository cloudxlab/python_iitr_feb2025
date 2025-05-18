# import lib

# p = float(input("Enter P: "))
# r = float(input("Enter R: "))
# t = float(input("Enter T: "))

# print("Interest: ", lib.interest(p, r, t))

# from mylib.lib import *
from mylib.legacy.lib import interest

intr = interest(100,3,6.0)
print(intr)

from mylib.liboops import Model
model = Model()
model.fit(1,2)

