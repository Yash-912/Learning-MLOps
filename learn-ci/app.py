import streamlit as st 

st.title("Power Calculator")
st.write("Enter a number whose power you want to calculate")

n=st.number_input("Enter your choice", value=1,step=1)
sq=n*n
cu=n**3
fi=n**5

st.write(f"The square of {n} is {sq}")
st.write(f"The cube of {n} is {cu}")
st.write(f"The fifth power of  {n} is {fi}")