from gensim.models.keyedvectors import KeyedVectors
import numpy as np
# 視覺化套件
import matplotlib
import matplotlib.pyplot as plt
# 主成分因子
from sklearn.decomposition import PCA
# 引入上述將文章斷詞後轉為300維向量的資料
word_vectors = KeyedVectors.load_word2vec_format("wiki300.model.bin", binary = True)
rawWordVec = word_vectors.vectors
print(rawWordVec.shape)
# 將原本300維向量空間降為2維
X_reduced = PCA(n_components=2).fit_transform(rawWordVec)
print(X_reduced.shape)
print(X_reduced[894787])
# 須先下載wqy-microhei.ttc，因中文顯示需做特殊處理
zhfont = matplotlib.font_manager.FontProperties(fname='./wqy-microhei.ttc')
# 畫圖
fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111)
index1 = word_vectors.wv.most_similar('裴秀智', topn=10)
index2 = word_vectors.wv.most_similar('楊冪', topn=10)
index1 = [word_vectors.vocab[word[0]].index for word in index1]
index2 = [word_vectors.vocab[word[0]].index for word in index2]
# add the index of center word
index1=np.append(index1,word_vectors.vocab['裴秀智'].index)
index2=np.append(index2,word_vectors.vocab['楊冪'].index)
for i in index1:
    ax.text(X_reduced[i][0],X_reduced[i][1],word_vectors.index2word[i], fontproperties = zhfont, color='C3')
for i in index2:
    ax.text(X_reduced[i][0],X_reduced[i][1],word_vectors.index2word[i], fontproperties = zhfont, color= 'C1')
"""
for i in index3:
    ax.text(X_reduced[i][0],X_reduced[i][1],model.vocab[i], fontproperties=zhfont,color='C7')
for i in index4:
    ax.text(X_reduced[i][0],X_reduced[i][1],model.vocab[i], fontproperties=zhfont,color='C0')
for i in index5:
    ax.text(X_reduced[i][0],X_reduced[i][1],model.vocab[i], fontproperties=zhfont,color='C4')
"""
plt.grid(b=True, which='major', color='#666666', linestyle='-')
ax.axis([-1.0,0.6,0.8,1.8])
plt.show()
plt.savefig("word2vector.png")
