#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      bhupendra
#
# Created:     26-06-2023
# Copyright:   (c) bhupendra 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------
"""
# Programme for building a calculator

a = int(input("Enter the first no :  "))
b = int(input("Enter the Second no : "))
operator = input("Enter the operator (+, -, *, /, %) : ")
if operator == "+" :
    print(a + b)
elif operator == "-" :
    print(a - b)
elif operator == "*" :
    print(a * b)
elif operator == "/" :
    print(a / b)
elif operator == "%" :
    print(a % b)
else:
    print("Invalid Input")
"""
"""
# Range and loop
i = 100
while i >= 0:
    print(i *  "*")
    i = i - 2

"""
"""
#range
for i in range(10):
    print(i)
"""
# Tuple
"""
students = ["Ram", "Shayam", "Kishan", "Radha", "Radhika"]
for student in students :
    if student == "Kishan":
        break;
    print(student)
"""
'''
students = ["Ram", "Shayam", "Kishan", "Radha", "Radhika"]
for student in students :
    if student == "Kishan":
        continue;
    print(student)
'''
"""
# List
marks = (90, 98, 99, 99, 99, 67)
print(marks.count(99))
print(marks.index(67))
"""
'''
#set
{}
#dictionary
marks = {"english" : 95, "chemistry" : 98}
print(marks["chemistry"])
marks["physics"] = 97;
print(marks)
'''
"""
#Functions
from math import sqrt
print(sqrt(256))
"""
'''#User defined function
def print_sum(first, second):
    print(first + second)

print_sum(100, 137718247)
'''
