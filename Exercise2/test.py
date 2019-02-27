import random

a={('a','b'):10,('a','c'):5,('a','e'):15,('b','d'):15}
# print(a.keys())
# for key in a.keys():
#     print(key[0])
#
# action_dict = {key[1] : value for key, value in a.items() if key[0] == 'a'}
# print(action_dict)
# print([action for action,value in action_dict.items() if value ==max(action_dict.values())])
# for i in range(10):
#     print([key for key, value in a.items() if value == max(a.values())])
#     print(random.choice([key for key, value in a.items() if value == max(a.values())]))
a[('a','b')]+=51
print(a)



