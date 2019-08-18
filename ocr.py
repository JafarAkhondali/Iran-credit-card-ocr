from skimage.feature import hog
from skimage import color
from sklearn.externals import joblib

knn = joblib.load('knn_model.pkl')

def predict_knn(df):
    predict = knn.predict(df.reshape(1,-1))[0]
    predict_proba = knn.predict_proba(df.reshape(1,-1))
    return predict, predict_proba[0][predict]
def inputdata(inputimage):
    (H, hogImage) = hog(
        inputimage,
        orientations=8,
        pixels_per_cell=(10, 10),
        cells_per_block=(7, 5),
        transform_sqrt=True,
        block_norm="L1",
        visualize=True
    )
    return predict_knn(H)