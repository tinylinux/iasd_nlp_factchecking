#Rearranging full.jsonl to better balance train/validation/test (labels are imbalanced)

f1=open("data/sufficient_facts/full.jsonl")
f2=open("data/sufficient_facts/full2.jsonl","w")

lines1=f1.readlines()


repeated=[]
irrelevant=[]
notenough=[]
for line in lines1:
    if "ENOUGH -- REPE" in line:
        repeated.append(line)
    elif "ENOUGH -- IRRE" in line:
        irrelevant.append(line)
    elif "NOT ENOUGH" in line:
        notenough.append(line)


train=[]
val=[]
test=[]

#Rebalance slightly the datasets
test=repeated[:5]+irrelevant[:20]+notenough[:25]
val=repeated[5:15]+irrelevant[20:120]+notenough[25:165]
train=repeated[15:]+irrelevant[120:400]+notenough[165:500]

print(len(test),len(val),len(train))

for t in test:
    f2.write(f'{t}')
for t in val:
    f2.write(f'{t}')
for t in train:
    f2.write(f'{t}')    