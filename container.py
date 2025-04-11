import streamlit as st

st.title("Mina Elgabalawy's Portfolio")
st.write("Data Storytelling")




tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])

with tab1:
    st.write('Tab 1')

with tab2:
    st.write('tab2')

with tab3:
    st.write('tab3')