ind2labels_dict = {
  0: "Background",
  1: "Hat",
  2: "Hair",
  3: "Sunglasses",
  4: "Upper-clothes",
  5: "Skirt",
  6: "Pants",
  7: "Dress",
  8: "Belt",
  9: "Left-shoe",
  10: "Right-shoe",
  11: "Face",
  12: "Left-leg",
  13: "Right-leg",
  14: "Left-arm",
  15: "Right-arm",
  16: "Bag",
  17: "Scarf"
}

# Calculate the reverse dict
labels2ind_dict = {v: k for k, v in ind2labels_dict.items()}


relevant_inds = [1,2,3,4,5,6,7,8,9,10,16,17]
