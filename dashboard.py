import streamlit as st
from supabase import create_client

SUPABASE_URL = "https://nwyzlwjvmbgyfiqcbcgz.supabase.co"
SUPABASE_KEY = "sb_publishable_NO9Q9eAW6aWRM-9D0Qipog_Y4c-2EgT"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.title("🚨 Fraud Detection Dashboard")

# Fetch data
data = supabase.table("Events").select("*").execute()

records = data.data

st.write("Total Events:", len(records))

# Show table
st.dataframe(records)