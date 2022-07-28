# %%
import os, sys, pathlib, json
import pandas as pd

# %% [markdown]
# Change labels into human-readable format

# %%
class_name_file = pd.read_csv('dataset/id_label_data.csv')

folder_dir = 'dataset/ILSVRC2012_img_val'

val_file_dir = pathlib.Path(folder_dir)
print(val_file_dir.exists())

type(class_name_file['origin'][0])

class_name_file.head()

# %%
for idx in range(len(class_name_file)):
  origin = class_name_file['origin'][idx]
  update = class_name_file['update'][idx]
  origin_path = pathlib.Path(folder_dir, origin)
  update_path = pathlib.Path(folder_dir, update)
  if update_path.exists():
    str_idx = str(idx)
    update_path = pathlib.Path(folder_dir, update+"_"+str_idx)
  os.rename(origin_path, update_path)

# %%


# %% [markdown]
# Saving ace results

# %%


# %%
name = 'windmill'
ace_dir = 'Ace-results2/ACE_Inception_'+name
#ace_dir = 'Ace-results2/ACE_DenseNet_'+name
#ace_dir = 'Ace-results2/ACE_Resnet_'+name
#ace_dir = 'Ace-results2/ACE_VGG_'+name

ace_file_dir = pathlib.Path(ace_dir)
print(ace_file_dir.exists())

target_class = 'windmill'

model = 'Inception'
#model = 'DenseNet'
#model= 'Resnet'
#model = 'VGG'

with open(ace_dir+"/"+target_class+"_"+model+"_tcav_scores.json") as f:
  tcav_data = json.load(f)

# %%
class_names = [name for name in os.listdir(ace_dir) if os.path.isdir(os.path.join(ace_dir, name))]
print(class_names)

concepts_dir = pathlib.Path(ace_dir, 'concepts')
masked_concepts_dir = pathlib.Path(ace_dir, 'masked_results')
results_dir = pathlib.Path(ace_dir, 'results')
summary_dir = pathlib.Path(ace_dir, 'results_summaries')
full_masked_dir = pathlib.Path(masked_concepts_dir, 'full_tcav_masked_results')

ace_data = dict()
ace_data['concept'] = dict()
ace_data['masked'] = dict()
ace_data['full_masked'] = dict()
ace_data['results'] = dict()
ace_data['summary'] = dict()

concept_names = [name for name in os.listdir(concepts_dir) if os.path.isdir(os.path.join(concepts_dir, name))]

for i in range(len(concept_names)):
  name = concept_names[i]
  if 'patches' not in name:
    continue
  each_file_dir = pathlib.Path(concepts_dir, name)
  file_lists = list(each_file_dir.glob("*.png"))
  files = []
  for file in file_lists:
    files.append(str(file))
  ace_data['concept'][name] = dict()
  ace_data['concept'][name]['file'] = files



# %%
masked_concept_names = [name for name in os.listdir(masked_concepts_dir) if os.path.isdir(os.path.join(masked_concepts_dir, name))]
masked_concept_names.remove("full_tcav_masked_results")
#masked_value = list(masked_concepts_dir.glob("*.json"))

'''if len(masked_value) != len(masked_concept_names):
  print("ERRORRR!!")
else:
  print("len(maksed_value) and len(masked_concept_names) are same")'''

for i in range(len(masked_concept_names)):
  name = str(masked_concept_names[i])
  concept_name = (name.split("_"))[-2]
  #mask_val = str(masked_value[i])
  '''if name not in mask_val:
    print("errror!!!")
    break'''
  each_file_dir = pathlib.Path(masked_concepts_dir, name)
  file_lists = list(each_file_dir.glob("*.png"))
  files = []
  for file in file_lists:
    files.append(str(file))
  ace_data['masked'][concept_name] = dict()
  ace_data['masked'][concept_name]['file'] = files
  #ace_data['masked'][concept_name]['masked_json'] = mask_val

result_tcav_png = list(full_masked_dir.glob("*_tcav.png"))
tcav_files = []
for file in result_tcav_png:
  tcav_files.append(str(file))

ace_data['full_masked'] = tcav_files

# %%
'''bn = list(tcav_data.keys())[0]
tcav_concepts = list(tcav_data[bn].keys())
len_concepts = list(ace_data['concept'].keys())

for idx in range(len(len_concepts)):
  con = tcav_concepts[idx]
  concept_data =  tcav_data[bn][con]
  avg_raw_tcav = concept_data["avg_raw_tcav"]
  for jdx in range(len(len_concepts)):
    if con+"_" in len_concepts[jdx]:
      #print(con)
      #print(len_concepts[jdx])
      ace_data['concept'][len_concepts[jdx]]["tcav_score"] = avg_raw_tcav'''


# %%
result_png = list(results_dir.glob("*.png"))
ace_data['results']['result_image'] = str(result_png[0])
print(result_png[0])

#raw_img_val = list(results_dir.glob("*.json"))
#ace_data['results']['raw_image'] = str(raw_img_val[0])

summary_txt = list(summary_dir.glob("*.txt"))
with open(str(summary_txt[0])) as file:
  lines = file.readlines()
ace_data['summary'] = lines

summary_json = list(summary_dir.glob("*.json"))
with open(str(summary_json[0])) as f:
  filename_dict = json.load(f)
ace_data['filename_match'] = filename_dict

# %%
with open("ace_json_results2/ace_"+model+"_"+target_class+".json", "w") as outfile:
  json.dump(ace_data, outfile, indent = 4)

# %%


# %%


# %%



