""" collection of various helper functions for running ACE"""

from multiprocessing import dummy as multiprocessing
import sys
import os, pathlib, json
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.lib.arraysetops import unique
from tcav_model import InceptionV3Wrapper_public, KerasModelWrapper
from tcav_model import *
#import tcav.model as model
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import tensorflow as tf
import seaborn as sns
from skimage import color
import cv2

def make_model(sess, model_to_run, model_path,
               labels_path, randomize=False,):
  #print("def make_model()")
  """Make an instance of a model.

  Args:
    sess: tf session instance.
    model_to_run: a string that describes which model to make.
    model_path: Path to models saved graph.
    randomize: Start with random weights
    labels_path: Path to models line separated class names text file.

  Returns:
    a model instance.

  Raises:
    ValueError: If model name is not valid.
  """
  if model_to_run == 'InceptionV3':
    print("model_to_run : InceptionV3")
    #mymodel = model.InceptionV3Wrapper_public(sess, model_saved_path=model_path, labels_path=labels_path)
    mymodel = InceptionV3Wrapper_public(sess, model_saved_path = model_path, labels_path = labels_path)
  elif model_to_run == 'GoogleNet':
    print("model_to_run : GoogleNet")
    # common_typos_disable
    #mymodel = model.GoogleNetWrapper_public(sess, model_saved_path=model_path, labels_path=labels_path)
    mymodel = GoogleNetWrapper_public(sess, model_saved_path=model_path, labels_path=labels_path)
  elif model_to_run == 'DenseNet':
    print("model_to_run : DenseNet")
    #mymodel = model.KerasModelWrapper(sess, model_path, labels_path)
    mymodel = KerasModelWrapper(sess, model_path, labels_path, model_to_run)
  elif model_to_run == 'Resnet':
    print("model_to_run : Resnet")
    mymodel = KerasModelWrapper(sess, model_path, labels_path, model_to_run)
  elif model_to_run == 'Inception':
    print("model_to_run : Inception")
    mymodel = KerasModelWrapper(sess, model_path, labels_path, model_to_run)
  elif model_to_run == 'VGG':
    print("model_to_run : VGG")
    mymodel = KerasModelWrapper(sess, model_path, labels_path, model_to_run)
  else:
    raise ValueError('Invalid model name')
  if randomize:  # randomize the network!
    sess.run(tf.global_variables_initializer())
  return mymodel


def load_image_from_file(filename, shape):
  #print("def load_image_from_file()")
  """Given a filename, try to open the file. If failed, return None.
  Args:
    filename: location of the image file
    shape: the shape of the image file to be scaled
  Returns:
    the image if succeeds, None if fails.
  Rasies:
    exception if the image was not the right shape.
  """
  if not tf.io.gfile.exists(filename):
    tf.logging.error('Cannot find file: {}'.format(filename))
    return None
  try:
    img = np.array(Image.open(filename).resize(
        shape, Image.BILINEAR))
    # Normalize pixel values to between 0 and 1.
    img = np.float32(img) / 255.0
    if not (len(img.shape) == 3 and img.shape[2] == 3):
      return None
    else:
      return img

  except Exception as e:
    #tf.logging.info(e)
    return None
  return img


def load_images_from_files(filenames, max_imgs=500, return_filenames=False,
                           do_shuffle=False, run_parallel=True,
                           shape=(299, 299),
                           num_workers=100):
  #print("def load_images_from_files()")
  """Return image arrays from filenames.
  Args:
    filenames: locations of image files.
    max_imgs: maximum number of images from filenames.
    return_filenames: return the succeeded filenames or not
    do_shuffle: before getting max_imgs files, shuffle the names or not
    run_parallel: get images in parallel or not
    shape: desired shape of the image
    num_workers: number of workers in parallelization.
  Returns:
    image arrays and succeeded filenames if return_filenames=True.
  """
  imgs = []
  # First shuffle a copy of the filenames.
  filenames = filenames[:]
  if do_shuffle:
    np.random.shuffle(filenames)
  if return_filenames:
    final_filenames = []
  if run_parallel:
    pool = multiprocessing.Pool(num_workers)
    imgs = pool.map(lambda filename: load_image_from_file(filename, shape),
                    filenames[:max_imgs])
    if return_filenames:
      final_filenames = [f for i, f in enumerate(filenames[:max_imgs])
                         if imgs[i] is not None]
    imgs = [img for img in imgs if img is not None]
  else:
    for filename in filenames:
      img = load_image_from_file(filename, shape)
      if img is not None:
        imgs.append(img)
        if return_filenames:
          final_filenames.append(filename)
      if len(imgs) >= max_imgs:
        break

  if return_filenames:
    return np.array(imgs), final_filenames
  else:
    return np.array(imgs)


def get_acts_from_images(imgs, model, bottleneck_name):
  #print("def get_acts_from_images()")
  """Run images in the model to get the activations.
  Args:
    imgs: a list of images
    model: a model instance
    bottleneck_name: bottleneck name to get the activation from
  Returns:
    numpy array of activations.
  """
  #print("model.run_examples(imgs, bottleneck_name): ", model.run_examples(imgs, bottleneck_name).shape)
  #print("np.asarray(model.run_examples(imgs, bottleneck_name)).squeeze(): ", np.asarray(model.run_examples(imgs, bottleneck_name)).squeeze().shape)
  return np.asarray(model.run_examples(imgs, bottleneck_name)).squeeze()


def flat_profile(cd, images, bottlenecks=None):
  #print("def flat_profile()")
  """Returns concept profile of given images.

  Given a ConceptDiscovery class instance and a set of images, and desired
  bottleneck layers, calculates the profile of each image with all concepts and
  returns a profile vector

  Args:
    cd: The concept discovery class instance
    images: The images for which the concept profile is calculated
    bottlenecks: Bottleck layers where the profile is calculated. If None, cd
      bottlenecks will be used.

  Returns:
    The concepts profile of input images using discovered concepts in
    all bottleneck layers.

  Raises:
    ValueError: If bottlenecks is not in right format.
  """
  profiles = []
  if bottlenecks is None:
    bottlenecks = list(cd.dic.keys())
  if isinstance(bottlenecks, str):
    bottlenecks = [bottlenecks]
  elif not isinstance(bottlenecks, list) and not isinstance(bottlenecks, tuple):
    raise ValueError('Invalid bottlenecks parameter!')
  for bn in bottlenecks:
    profiles.append(cd.find_profile(str(bn), images).reshape((len(images), -1)))
  profile = np.concatenate(profiles, -1)
  return profile


def cross_val(a, b, methods):
  #print("def cross_val()")
  """Performs cross validation for a binary classification task.

  Args:
    a: First class data points as rows
    b: Second class data points as rows
    methods: The sklearn classification models to perform cross-validation on

  Returns:
    The best performing trained binary classification odel
  """
  x, y = binary_dataset(a, b)
  best_acc = 0.
  if isinstance(methods, str):
    methods = [methods]
  best_acc = 0.
  for method in methods:
    temp_acc = 0.
    params = [10**e for e in [-4, -3, -2, -1, 0, 1, 2, 3]]
    for param in params:
      clf = give_classifier(method, param)
      acc = cross_val_score(clf, x, y, cv=min(100, max(2, int(len(y) / 10))))
      if np.mean(acc) > temp_acc:
        temp_acc = np.mean(acc)
        best_param = param
    if temp_acc > best_acc:
      best_acc = temp_acc
      final_clf = give_classifier(method, best_param)
  final_clf.fit(x, y)
  return final_clf, best_acc


def give_classifier(method, param):
  #print("def give_classifier()")
  """Returns an sklearn classification model.

  Args:
    method: Name of the sklearn classification model
    param: Hyperparameters of the sklearn model

  Returns:
    An untrained sklearn classification model

  Raises:
    ValueError: if the model name is invalid.
  """
  if method == 'logistic':
    return linear_model.LogisticRegression(C=param)
  elif method == 'sgd':
    return linear_model.SGDClassifier(alpha=param)
  else:
    raise ValueError('Invalid model!')


def binary_dataset(pos, neg, balanced=True):
  #print("def binary_dataset()")
  """Creates a binary dataset given instances of two classes.

  Args:
     pos: Data points of the first class as rows
     neg: Data points of the second class as rows
     balanced: If true, it creates a balanced binary dataset.

  Returns:
    The data points of the created data set as rows and the corresponding labels
  """
  if balanced:
    min_len = min(neg.shape[0], pos.shape[0])
    ridxs = np.random.permutation(np.arange(2 * min_len))
    x = np.concatenate([neg[:min_len], pos[:min_len]], 0)[ridxs]
    y = np.concatenate([np.zeros(min_len), np.ones(min_len)], 0)[ridxs]
  else:
    ridxs = np.random.permutation(np.arange(len(neg) + len(pos)))
    x = np.concatenate([neg, pos], 0)[ridxs]
    y = np.concatenate(
        [np.zeros(neg.shape[0]), np.ones(pos.shape[0])], 0)[ridxs]
  return x, y

from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def apply_mask_on_image(image, mask, b_mask, color_list, opacity):
  image_mask = color.label2rgb(mask, colors = color_list, bg_label= 0)
  masked_applied_image = image * (1 - opacity) + image_mask * opacity
  #masked_applied_image = ((masked_applied_image * 255).clip(0, 255).round().astype(np.uint8))
  img_masked_applied = mark_boundaries(masked_applied_image, b_mask, color = (1,1,0), mode = 'inner')
  return img_masked_applied

def save_mask_plot(cd, bn, address, raw_img_address, tcav_scores):
  print("def save_mask_plot()")
  concepts = cd.dic[bn]['concepts']

  print("cd.discovery_images num: ", len(cd.discovery_images))

  raw_img_dict = dict()
  raw_img_dict['raw_img_value'] = dict()

  for idx in range(len(cd.discovery_images)):
    raw_img = cd.discovery_images[idx]
    image_id = '0' * int(np.ceil(3 - np.log10(idx + 1))) + '{}'.format(idx + 1)
    raw_img_dict['raw_img_value'][image_id] = raw_img.tolist()

  #with open(raw_img_address+"raw_img_vals.json", "w") as outfile:
  #  json.dump(raw_img_dict, outfile, indent = 4)

  for n, concept in enumerate(concepts):
    concept_images = cd.dic[bn][concept]['images']
    concept_patches = cd.dic[bn][concept]['patches']
    concept_image_numbers = cd.dic[bn][concept]['image_numbers']

    boundaries_dir = os.path.join(address, bn + '_' + concept + '_boundaries')

    #mask_dict = dict()
    #mask_dict['mask_value'] = dict()
    tf.io.gfile.makedirs(boundaries_dir)
    for idx in range(len(concept_images)):
      #image_name = '0' * int(np.ceil(2 - np.log10(idx + 1))) + '{}_{}'.format(idx + 1, concept_image_numbers[idx])
      image_name = '0' * int(np.ceil(3 - np.log10(concept_image_numbers[idx] + 1))) + '{}_{}'.format(concept_image_numbers[idx] + 1, idx + 1)
      mask = 1 - (np.mean(concept_patches[idx] == float(cd.average_image_value) / 255, -1) == 1)
      image = cd.discovery_images[concept_image_numbers[idx]]
      #if idx == 0:
      #  mask_dict['mask_size'] = mask.shape
      #  mask_dict['image_size'] = image.shape
      #mask_dict['mask_value'][image_name] = mask.tolist()
      boundary_image = mark_boundaries(image, mask, color =(1,1,0), mode = 'thick')
      file_address = os.path.join(boundaries_dir, image_name + '.png')
      plt.imsave(file_address, boundary_image)

    #with open(boundaries_dir+"_masked_value.json", "w") as outfile:
    #  json.dump(mask_dict, outfile, indent = 4)

  avg_tcav_scores = []
  patch_masks = cd.patch_masks_dict
  print("cd.patch_masks_dict num: ", len(cd.patch_masks_dict))
  for n, concept in enumerate(concepts):
    concept_num = concept.split("_")[-1]
    concept_patch_masks = cd.dic[bn][concept]['masks']
    #concept_tcav_score = cd.tcav_scores[bn][concept]['avg_raw_tcav']
    concept_avg_tcav_score = np.mean(tcav_scores[bn][concept]['mean_tcav'])
    avg_tcav_scores.append(concept_avg_tcav_score)

    '''if len(concept_patch_masks) > len(concept_tcav_score):
      concept_patch_length = len(concept_tcav_score)
    else:
      concept_patch_length = len(concept_patch_masks)'''
    for idx in range(len(concept_patch_masks)):
      patch_id = concept_patch_masks[idx]
      #tcav = concept_tcav_score[idx]
      iid, fn = patch_id.split(":")
      #patch_masks[iid][int(fn)][2] = tcav
      patch_masks[iid][int(fn)][2] = concept_avg_tcav_score
      patch_masks[iid][int(fn)][3] = concept_num

  img_id_list = list(patch_masks.keys())
  for idx in range(len(img_id_list)):
    iid = img_id_list[idx]
    raw_img = np.array(raw_img_dict['raw_img_value'][iid])
    pm_list = patch_masks[iid]
    tcav_mask = np.zeros(shape = pm_list[0][1].shape)
    tcav_mask_num_order = np.zeros(shape = pm_list[0][1].shape)
    boundary_mask = np.zeros(shape = pm_list[0][1].shape)
    cp_num = 0
    x_pos_list = []
    y_pos_list = []
    tcav_val_list = []
    adjusted_tcav_list = []
    for jdx in range(len(pm_list)):
      patch_id, img_masks, tcav, concept_num = pm_list[jdx]
      boundary_mask += (patch_id+1) * img_masks
      if tcav != -1:

      #if tcav > np.median(avg_tcav_scores):
        cp_num += 1
        arr = np.argwhere(img_masks != 0)
        y_arr, x_arr = np.split(arr, 2, axis = 1)
        x_pos_list.append(np.average(x_arr))
        y_pos_list.append(np.average(y_arr))
        round_tcav_score = str("{0:.3f}".format(float(tcav)))
        tcav_val_list.append(concept_num +": "+round_tcav_score)

        adjusted_tcav = int(tcav * 100)
        adjusted_tcav_list.append(adjusted_tcav)
        tcav_mask += adjusted_tcav * img_masks
        tcav_mask_num_order += adjusted_tcav * img_masks
        #print("id - tcav*100 :", adjusted_tcav)

      else:
        tcav_mask += 0 * img_masks
        tcav_mask_num_order += 0 * img_masks

    full_tcav_dir = os.path.join(address, 'full_tcav_masked_results')
    tf.io.gfile.makedirs(full_tcav_dir)
    mask_order = 1
    adjusted_tcav_list.sort()
    for ldx in range(len(adjusted_tcav_list)):
      tval = adjusted_tcav_list[ldx]
      tcav_mask_num_order = np.where(tcav_mask_num_order == tval, mask_order, tcav_mask_num_order)
      mask_order += 1
    #color_list = sns.color_palette("Greys_r", cp_num+1)

    unique_tcav_mask = np.unique(tcav_mask)
    color_list = sns.color_palette("Reds", len(unique_tcav_mask) - 1)
    #color_list.reverse()
    #sorted_color_list = {order : color_list for order, color_list in sorted(zip(unique_tcav_mask, color_list))}
    mask_applied_image = apply_mask_on_image(raw_img, tcav_mask_num_order, boundary_mask, color_list, opacity = 0.4)
    fig = plt.figure()
    plt.axis("off")
    plt.imshow(mask_applied_image)
    for idx in range(len(tcav_val_list)):
      plt.text(x = x_pos_list[idx] - 8, y = y_pos_list[idx], s = str(tcav_val_list[idx]), color = "white", fontsize = 25, fontweight = 'bold')

    #file_address2 = os.path.join(full_tcav_dir, iid + '_mask.png')
    file_address3 = os.path.join(full_tcav_dir, iid + '_tcav.png')

    #plt.imsave(file_address2, mask_applied_image)
    plt.savefig(file_address3, pad_inches = 0, bbox_inches = 'tight')
    plt.close()

    '''mask_applied_image2 = apply_mask_on_image(raw_img, tcav_mask_num_order, boundary_mask, color_list, opacity = 1)
    fig = plt.figure()
    plt.axis("off")
    plt.imshow(mask_applied_image2)
    for idx in range(len(tcav_val_list)):
      plt.text(x = x_pos_list[idx], y = y_pos_list[idx], s = str(tcav_val_list[idx]), color = "white", fontsize = 15, fontweight = 'bold')
    file_address4 = os.path.join(full_tcav_dir, iid + '_full.png')

    plt.savefig(file_address4, pad_inches = 0, bbox_inches = 'tight')'''

  #with open(raw_img_address+"mask_values.json", "w") as outfile:
  #  json.dump(patch_masks, outfile, indent = 4)

def plot_concepts(cd, bn, num=10, address=None, mode='diverse', concepts=None):
  #print("def plot_concepts()")
  """Plots examples of discovered concepts.

  Args:
    cd: The concept discovery instance
    bn: Bottleneck layer name
    num: Number of images to print out of each concept
    address: If not None, saves the output to the address as a .PNG image
    mode: If 'diverse', it prints one example of each of the target class images
      is coming from. If 'radnom', randomly samples exmples of the concept. If
      'max', prints out the most activating examples of that concept.
    concepts: If None, prints out examples of all discovered concepts.
      Otherwise, it should be either a list of concepts to print out examples of
      or just one concept's name

  Raises:
    ValueError: If the mode is invalid.
  """
  if concepts is None:
    concepts = cd.dic[bn]['concepts']
  elif not isinstance(concepts, list) and not isinstance(concepts, tuple):
    concepts = [concepts]
  num_concepts = len(concepts)
  plt.rcParams['figure.figsize'] = num * 2.1, 4.3 * num_concepts
  fig = plt.figure(figsize=(num * 2, 4 * num_concepts))
  outer = gridspec.GridSpec(num_concepts, 1, wspace=0., hspace=0.3)
  for n, concept in enumerate(concepts):
    inner = gridspec.GridSpecFromSubplotSpec(
        2, num, subplot_spec=outer[n], wspace=0, hspace=0.1)
    concept_images = cd.dic[bn][concept]['images']
    concept_patches = cd.dic[bn][concept]['patches']
    concept_image_numbers = cd.dic[bn][concept]['image_numbers']
    if mode == 'max':
      idxs = np.arange(len(concept_images))
    elif mode == 'random':
      idxs = np.random.permutation(np.arange(len(concept_images)))
    elif mode == 'diverse':
      idxs = []
      while True:
        seen = set()
        for idx in range(len(concept_images)):
          if concept_image_numbers[idx] not in seen and idx not in idxs:
            seen.add(concept_image_numbers[idx])
            idxs.append(idx)
        if len(idxs) == len(concept_images):
          break
    else:
      raise ValueError('Invalid mode!')
    idxs = idxs[:num]
    for i, idx in enumerate(idxs):
      ax = plt.Subplot(fig, inner[i])
      ax.imshow(concept_images[idx])
      ax.set_xticks([])
      ax.set_yticks([])
      if i == int(num / 2):
        ax.set_title(concept)
      ax.grid(False)
      fig.add_subplot(ax)
      ax = plt.Subplot(fig, inner[i + num])
      #여기서 mask 계산하고 mark boundaries를 하나보다
      mask = 1 - (np.mean(concept_patches[idx] == float(
          cd.average_image_value) / 255, -1) == 1)
      image = cd.discovery_images[concept_image_numbers[idx]]
      ax.imshow(mark_boundaries(image, mask, color=(1, 1, 0), mode='thick'))
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_title(str(concept_image_numbers[idx]))
      ax.grid(False)
      fig.add_subplot(ax)
  plt.suptitle(bn)
  if address is not None:
    with tf.io.gfile.GFile(address + bn + '.png', 'w') as f:
      fig.savefig(f)
    plt.clf()
    plt.close(fig)


def cosine_similarity(a, b):
  #print("def cosine_similarity()")
  """Cosine similarity of two vectors."""
  assert a.shape == b.shape, 'Two vectors must have the same dimensionality'
  a_norm, b_norm = np.linalg.norm(a), np.linalg.norm(b)
  if a_norm * b_norm == 0:
    return 0.
  cos_sim = np.sum(a * b) / (a_norm * b_norm)
  return cos_sim


def similarity(cd, num_random_exp=None, num_workers=25):
  #print("def similarity()")
  """Returns cosine similarity of all discovered concepts.

  Args:
    cd: The ConceptDiscovery module for discovered conceps.
    num_random_exp: If None, calculates average similarity using all the class's
      random concepts. If a number, uses that many random counterparts.
    num_workers: If greater than 0, runs the function in parallel.

  Returns:
    A similarity dict in the form of {(concept1, concept2):[list of cosine
    similarities]}
  """

  def concepts_similarity(cd, concepts, rnd, bn):
    #print("def concepts_similarity()")
    """Calcualtes the cosine similarity of concept cavs.

    This function calculates the pairwise cosine similarity of all concept cavs
    versus an specific random concept

    Args:
      cd: The ConceptDiscovery instance
      concepts: List of concepts to calculate similarity for
      rnd: a random counterpart
      bn: bottleneck layer the concepts belong to

    Returns:
      A dictionary of cosine similarities in the form of
      {(concept1, concept2): [list of cosine similarities], ...}
    """
    similarity_dic = {}
    for c1 in concepts:
      cav1 = cd.load_cav_direction(c1, rnd, bn)
      for c2 in concepts:
        if (c1, c2) in similarity_dic.keys():
          continue
        cav2 = cd.load_cav_direction(c2, rnd, bn)
        similarity_dic[(c1, c2)] = cosine_similarity(cav1, cav2)
        similarity_dic[(c2, c1)] = similarity_dic[(c1, c2)]
    return similarity_dic

  similarity_dic = {bn: {} for bn in cd.bottlenecks}
  if num_random_exp is None:
    num_random_exp = cd.num_random_exp

  #randoms = ['random500_{}'.format(i) for i in np.arange(num_random_exp)]
  randoms = [name for name in os.listdir(str(cd.source_dir)) if os.path.isdir(os.path.join(str(cd.source_dir), name))]
  randoms.remove(str(cd.target_class))
  randoms.remove("random_discovery")
  concepts = {}
  for bn in cd.bottlenecks:
    concepts[bn] = [cd.target_class, cd.random_concept] + cd.dic[bn]['concepts']
  for bn in cd.bottlenecks:
    concept_pairs = [(c1, c2) for c1 in concepts[bn] for c2 in concepts[bn]]
    similarity_dic[bn] = {pair: [] for pair in concept_pairs}
    def t_func(rnd):
      return concepts_similarity(cd, concepts[bn], rnd, bn)
    if num_workers:
      pool = multiprocessing.Pool(num_workers)
      sims = pool.map(lambda rnd: t_func(rnd), randoms)
    else:
      sims = [t_func(rnd) for rnd in randoms]
    while sims:
      sim = sims.pop()
      for pair in concept_pairs:
        similarity_dic[bn][pair].append(sim[pair])
  return similarity_dic


def save_ace_report(cd, accs, scores, address):
  #print("def save_ace_report()")
  """Saves TCAV scores.

  Saves the average CAV accuracies and average TCAV scores of the concepts
  discovered in ConceptDiscovery instance.

  Args:
    cd: The ConceptDiscovery instance.
    accs: The cav accuracy dictionary returned by cavs method of the
      ConceptDiscovery instance
    scores: The tcav score dictionary returned by tcavs method of the
      ConceptDiscovery instance
    address: The address to save the text file in.
  """
  report = '\n\n\t\t\t ---CAV accuracies---'
  for bn in cd.bottlenecks:
    report += '\n'
    for concept in cd.dic[bn]['concepts']:
      report += '\n' + bn + ':' + concept + ':' + str(
          np.mean(accs[bn][concept]))
  with tf.io.gfile.GFile(address, 'w') as f:
    f.write(report)
  #report = '\n\n\t\t\t ---TCAV scores---'
  report = ''
  for bn in cd.bottlenecks:
    #report += '\n'
    for concept in cd.dic[bn]['concepts']:
      pvalue = cd.do_statistical_testings(scores[bn][concept]['mean_tcav'], scores[bn][cd.random_concept]['mean_tcav'])
      concept_name = (str(concept).split("_"))[-1]
      report += '{}:{}:{},{}\n'.format(bn, str(concept_name), np.mean(scores[bn][concept]['mean_tcav']), pvalue)
      #report += '\n{}:{}:{},{}'.format(bn, concept, np.mean(scores[bn][concept]), pvalue)
  with tf.io.gfile.GFile(address, 'w') as f:
    f.write(report)


def save_concepts(cd, concepts_dir):
  print("def save_concepts()")
  """Saves discovered concept's images or patches.

  Args:
    cd: The ConceptDiscovery instance the concepts of which we want to save
    concepts_dir: The directory to save the concept images
  """
  for bn in cd.bottlenecks:
    for concept in cd.dic[bn]['concepts']:
      patches_dir = os.path.join(concepts_dir, bn + '_' + concept + '_patches')
      images_dir = os.path.join(concepts_dir, bn + '_' + concept)
      patches = (np.clip(cd.dic[bn][concept]['patches'], 0, 1) * 256).astype(
          np.uint8)
      images = (np.clip(cd.dic[bn][concept]['images'], 0, 1) * 256).astype(
          np.uint8)
      tf.io.gfile.makedirs(patches_dir)
      tf.io.gfile.makedirs(images_dir)
      image_numbers = cd.dic[bn][concept]['image_numbers']
      image_addresses, patch_addresses = [], []
      for i in range(len(images)):
        #image_name = '0' * int(np.ceil(2 - np.log10(i + 1))) + '{}_{}'.format(i + 1, image_numbers[i])
        image_name = '0' * int(np.ceil(3 - np.log10(image_numbers[i] + 1))) + '{}_{}'.format(image_numbers[i] + 1, i + 1)
        patch_addresses.append(os.path.join(patches_dir, image_name + '.png'))
        image_addresses.append(os.path.join(images_dir, image_name + '.png'))
      save_images(patch_addresses, patches)
      save_images(image_addresses, images)

def save_images_filename(cd, address):
  raw_image_file_names = cd.discovery_images_filename
  image_file_names = dict()
  for i, file_names in enumerate(raw_image_file_names):
    image_name = '0' * (3 - int(np.log10(i + 1))) + str(i + 1)
    image_file_names[file_names] = image_name

  #image_filename_address = os.path.join(address, 'raw_ace_filename_match.json')
  with open(address, "w") as outfile:
    json.dump(image_file_names, outfile, indent = 4)


def save_images(addresses, images):
  #print("def save_images()")
  """Save images in the addresses.

  Args:
    addresses: The list of addresses to save the images as or the address of the
      directory to save all images in. (list or str)
    images: The list of all images in numpy uint8 format.
  """
  if not isinstance(addresses, list):
    image_addresses = []
    for i, image in enumerate(images):
      image_name = '0' * (3 - int(np.log10(i + 1))) + str(i + 1) + '.png'
      image_addresses.append(os.path.join(addresses, image_name))
    addresses = image_addresses
  assert len(addresses) == len(images), 'Invalid number of addresses'
  for address, image in zip(addresses, images):
    with tf.io.gfile.GFile(address, 'w') as f:
      Image.fromarray(image).save(f, format='PNG')

