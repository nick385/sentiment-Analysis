import streamlit as st
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.externals import joblib
import joblib
from PIL import Image
import plotly.figure_factory as ff

z = [[0.1, .7],
     [0.7, .2]]
     

x = ['neg','pos']
y = ['neg','pos']

z_text = [ ['160', '28'],
           ['33', '167']
          ]

fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis',
                       )


pipeline = joblib.load('transform_predict.joblib')

def predict(data):
    return pipeline.predict([data])[0]




def main():
    st.title("Sentiment Analysis Tool")
    st.subheader("Enter Your Text")
    message = st.text_area("","Type Here")
    if st.button("Analyze"):
        result = predict(message)
        if result=='pos':
            st.success('Positive')
        else:
            st.warning('Negative')
    st.subheader("Confusion Matrix")
    st.plotly_chart(fig)
    
   









if __name__ =='__main__':
    main()