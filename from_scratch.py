import pandas as pd
import numpy as np

df = pd.read_csv('student-mat.csv',sep=';')

print(df)
print(df['G3'].value_counts())
print(df['G3'].head())

#print(df['target'].value_counts())
print(df.columns)
print(df.info())

features = [
    "age", "Medu", "Fedu",
    "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout",
    "Dalc", "Walc", "health",
    "absences"]

X = df[features]
y = (df["G3"] >= 10).astype(int).values

np.random.seed(42)

indices = np.random.permutation(len(X))
test_size = int(len(X)*0.2)

train_indices = indices[test_size:]
test_indices = indices[:test_size]

x_train = X.iloc[train_indices]
x_test = X.iloc[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

def weight(g,h,l):
   G = np.sum(g)
   H = np.sum(h)
   w = -G/(H+1)
   return w

def find_thresholds(x):
   thresholds = []
   for feature_index in range(len(x[0])):
      feature_values = []
      for i in range(len(x)):
         feature_values.append(x[i][feature_index])
      thres = []
      for i in range(len(feature_values)-1):
         threshold = (feature_values[i]+feature_values[i+1]) / 2
         thres.append(threshold)
      thresholds.append(thres)
   return thresholds

def split_data(x,g,h,feature_index,threshold):
   x_left = []
   x_right = []
   g_right = []
   g_left = []
   h_right = []
   h_left = []
   for i in range(len(x)):
    if x[i][feature_index] <= threshold:
       x_left.append(x[i])
       g_left.append(g[i])
       h_left.append(h[i])
    else:
       x_right.append(x[i])
       g_right.append(g[i])
       h_right.append(h[i])
   return x_left,x_right,g_left,g_right,h_left,h_right

def find_gain(y,g_left,h_left,g_right,h_right,l,p):
   pred = np.mean(y)
   left_g = np.sum(g_left)
   right_g = np.sum(g_right)
   left_h = np.sum(h_left)
   right_h = np.sum(h_right)
   parent_g = np.sum(pred - y)
   parent_h = np.ones(len(y))
   gain = 0.5*((((left_g)**2/(left_h+l)) + (right_g)**2/(right_h+l))- parent_g / (parent_h+l)) - p
   return np.max(gain)

def best_split(x,y,g,h,l,p,thresholds):
   best_gain = float('-inf')
   best_feature = 0
   best_threshold = 0
   best_x_left = None
   best_x_right = None
   best_g_left = None
   best_g_right = None
   best_h_left = None
   best_h_right = None
   for feature_index in range(len(x[0])):
      for threshold in thresholds[feature_index]:
         x_left,x_right,g_left,g_right,h_left,h_right = split_data(x,g,h,feature_index,threshold)
         gain = find_gain(y,g_left,h_left,g_right,h_right,l,p)
         if gain > best_gain:
            best_gain = gain
            best_feature = feature_index
            best_threshold = threshold
            best_x_left = x_left
            best_x_right = x_right
            best_g_left = g_left
            best_g_right = g_right
            best_h_left = h_left
            best_h_right = h_right
   return best_gain,best_feature,best_threshold,best_x_left,best_x_right,best_g_left,best_g_right,best_h_left,best_h_right

def build_tree(x,g,h,l,p,depth=0,max_depth=4):
   if depth >= max_depth:
      return weight(g,h,l)
   thresholds = find_thresholds(x)
   best_gain,best_feature,best_threshold,best_x_left,best_x_right,best_g_left,best_g_right,best_h_left,best_h_right = best_split(x,y,g,h,l,p,thresholds)
   if best_gain <= 0:
      return weight(g,h,l)
   if best_g_left is None:
      return weight(g,h,l)
   left_subtree = build_tree(best_x_left,best_g_left,best_h_left,l,p,depth+1,max_depth)
   right_subtree = build_tree(best_x_right,best_g_right,best_h_right,l,p,depth+1,max_depth)
   return best_feature,best_threshold,left_subtree,right_subtree

def predict(sample,tree):
  if isinstance(tree,(float,np.floating)):
    return tree
  best_feature,best_threshold,left_subtree,right_subtree = tree
  if sample[best_feature] <= best_threshold:
    return predict(sample,left_subtree)
  else:
    return predict(sample,right_subtree)

def predictions(x,tree):
  prediction = []
  for sample in x:
    prediction.append(predict(sample,tree))
  return prediction

lr = 0.1
def xgboost(x,y,l=1,p=1):
   trees = []
   pred = np.zeros(len(y))
   prob = 1 / (1+np.exp(-pred))
   g = prob - y
   h = prob*(1-prob)
   for _ in range(50):
      prob = 1 / (1+np.exp(-pred))
      g = prob - y
      h = prob*(1-prob)
      tree = build_tree(x,g,h,l,p)
      tree_pred = np.array(predictions(x,tree))
      pred += lr*tree_pred
      trees.append(tree)
   return trees

def test_pred(x_test,x,y):
   pred = np.zeros(len(x_test))
   trees = xgboost(x,y)
   for tree in trees:
      tree_pred = np.array(predictions(x_test,tree))
      pred += lr*tree_pred
   return pred
raw_pred = test_pred(x_test,x_train,y_train)

y_pred = []
for i in raw_pred:
  if i <= 0.5:
    y_pred.append(0)
  else:
    y_pred.append(1)


c = 0
for i in range(len(y_test)):
  if y_pred[i] == y_test[i]:
    c += 1
accuracy = c / len(y_test)


tp = 0
tn = 0
fp = 0
fn = 0
for i in range(len(y_test)):
   if y_pred[i] == 1 and y_test[i] == 1:
      tp += 1
   elif y_pred[i] == 0 and y_test[i] == 0:
      tn += 1
   elif y_test[i] == 0 and y_pred[i] == 1:
      fp += 1
   else:
      fn += 1
precision = (tp / (tp+fp))
recall = (tp/(tp+fn))

print('Precision:',precision)
print('recall:',recall)

def confusion_matrix(y_pred,y_test):
   classes = np.unique(y_test)
   matrix = np.zeros((len(classes),len(classes)),dtype=int)
   for predicted, actual in zip(y_pred,y_test):
      actual_index = np.where(classes == actual)[0][0]
      predicted_index = np.where(classes == predicted)[0][0]

      matrix[actual_index][predicted_index] += 1
   return matrix

print('accuracy:',accuracy*100)
print('Precision:',precision)
print('recall:',recall)
print('Confusion Matrix:',confusion_matrix(y_pred,y_test))
      


