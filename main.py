import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


st.set_page_config(layout="wide",
                   page_title="INDIAN SCHOOL EDUCATION STATISTICS ANALYSIS APP",
                   page_icon='📚')

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showPyplotGlobalUse', False)



rename_dict={"Primary_Total": "Primary", 
             "Upper Primary_Total": "Upper Primary", 
             "Secondary _Total": "Secondary", 
             "HrSecondary_Total": "Higher Secondary"}


@st.cache_data()
def load_data():
    df = pd.read_csv('dropout-ratio-2012-2015.csv')
    india = gpd.read_file('Indian_States.txt')
    df.replace("NR", np.nan, inplace=True)
    df.rename(columns=rename_dict, inplace=True)
    df["Primary"] = df["Primary"].astype(float)
    df["Upper Primary"] = df["Upper Primary"].astype(float)
    df["Secondary"] = df["Secondary"].astype(float)
    df["Higher Secondary"] = df["Higher Secondary"].astype(float)
    df_total = df[['Primary','Upper Primary','Secondary','Higher Secondary']]
    df["Average"] = df_total.mean(axis=1)


    return df,india, 
 

with st.spinner("Processing Dropout data...."):
    df, india = load_data()

num_cols = df.select_dtypes(include='number').columns
cat_cols = df.select_dtypes(include='object').columns



st.markdown(
    """<h1 style="text-align: center; font-size: 60px;">Indian School Education Statistics Analysis</h1>""", 
    unsafe_allow_html=True
)
c1, c2,c3 = st.columns([1,6,1])
c2.image('0922bb92-5a3a-4539-9dc5-588cab4dc4a3.webp', use_column_width=True)
st.markdown("""**Data source:** [Indian School Education Statistics](https://data.gov.in/resources/stateut-wise-average-annual-drop-out-rate-2012-13-2014-15-ministry-human-resource
""")




st.markdown("""<h1 style="text-align: center;">Data Analysis and Visualization</h1>""", unsafe_allow_html=True)
analysis_type = st.selectbox(
    "Select the type of analysis",
    options=["DROPOUT", "ENROLLMENT", "STATES WITH FACILITIES"]
)

if analysis_type == "DROPOUT":

    st.markdown("""<h2 style="text-align: center;">Dropout Analysis and Visualization</h2>""", unsafe_allow_html=True)
    states = df["State_UT"].unique()
    # st.write(states.tolist())
    c1, c2 = st.columns(2)
    c1.subheader("Select the state to visualize the dropout rates")
    state = c1.selectbox("State", states)
    col = c1.selectbox("Select the level to visualize the dropout rates", num_cols)
    df_state = df[df["State_UT"] == state]
    c1.dataframe(df_state, use_container_width=True)
    c2.subheader("Line chart of dropout rates for each level")


    fig = px.bar(
        data_frame=df_state,
        x="year",
        y=col,
        title=f"Dropout Rate for {col} Level in {state}",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Alphabet,
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=f"Dropout Rate (%)",
        margin=dict(t=50, b=50, l=50, r=50),
        
    )

    # Display the plot using Streamlit
    c2.plotly_chart(fig, use_container_width=True)



    t1, t2 = st.tabs(["Bivariate","Trivariate"])
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    with t1:
        c1, c2 = st.columns(2)
        col1 = c1.radio("Select the first column for scatter plot", num_cols,)
        col2 = c2.radio("Select the Second column for scatter plot", num_cols)
        fig = px.scatter(df, x=col1, y=col2, title=f'{col1} vs {col2}',color_discrete_sequence=px.colors.diverging.Temps)
        st.plotly_chart(fig, use_container_width=True)
        
    with t2:
        c1, c2, c3 = st.columns(3)
        col1 = c1.selectbox("Select the first column for 3d plot", num_cols)
        col2 = c2.selectbox("Select the second column for 3d plot", num_cols)
        col3 = c3.selectbox("Select the third column for 3d plot", num_cols)
        fig = px.scatter_3d(df, x=col1, y=col2,
                            z=col3, title=f'{col1} vs {col2} vs {col3}',
                            height=700)
        st.plotly_chart(fig, use_container_width=True)
        

    # Create tabs
    tabs = st.tabs(["Best States", "Worst States","Gender Comparison"])

    # Best states tab
    with tabs[0]:
        st.subheader('Top 5 States with the Lowest Dropout Rates')
        df_best10 = df.sort_values(by="Average", ascending=True).head(8)  # Changed to top 10
        fig_best = px.pie(
            df_best10,
            values="Average",
            names="State_UT",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.6,
        )

        fig_best.add_annotation(
            x=0,
            y=0,
            showarrow=False,
            font=dict(size=15, color="black"),
            xanchor="center",
            yanchor="middle",
        )
        fig_best.update_traces(textposition="inside", textinfo="label")
        fig_best.update_layout(
            margin=dict(t=50, b=50, l=50, r=50),
        )

        st.plotly_chart(fig_best, use_container_width=True)

    # Worst states tab
    with tabs[1]:
        st.subheader('Top 5 States with the Highest Dropout Rates')
        df_worst5 = df.sort_values(by="Average", ascending=False).head(5)
        fig_worst = px.pie(
            df_worst5,
            values="Average",
            names="State_UT",
            color_discrete_sequence=px.colors.carto.Darkmint,
            hole=0.6,
        )

    

        # Add annotation and update layout
        fig_worst.add_annotation(
            x=0,
            y=0,
            showarrow=False,
            font=dict(size=15, color="black"),
            xanchor="center",
            yanchor="middle",
        )
        fig_worst.update_traces(textposition="inside", textinfo="label")
        fig_worst.update_layout(
            margin=dict(t=50, b=50, l=50, r=50),
            
        )
        
        st.plotly_chart(fig_worst, use_container_width=True)

    with tabs[2]:
        st.subheader('Comparison of Dropout Rates between Boys and Girls')
        df_boys=df[["Primary_Boys","Upper Primary_Boys","Secondary _Boys","HrSecondary_Boys"]]
        df_girls=df[["Primary_Girls","Upper Primary_Girls","Secondary _Girls","HrSecondary_Girls"]]
        df_boys = df_boys.apply(pd.to_numeric, errors='coerce')
        df_girls=df_girls.apply(pd.to_numeric, errors='coerce')
        boys_total = df_boys.sum().sum()
        girls_total = df_girls.sum().sum()
        boys_percentage = (boys_total / (boys_total + girls_total)) * 100
        girls_percentage = (girls_total / (boys_total + girls_total)) * 100
        labels = ['Boys', 'Girls']
        sizes = [boys_percentage, girls_percentage]
        colors =px.colors.qualitative.Vivid
        fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, textinfo='label+percent',
                                marker=dict(colors=colors), hole=0.6)])

        fig.update_layout(
            margin=dict(t=50, b=50, l=50, r=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    df['year'] = pd.to_numeric(df['year'].str[:4], errors='coerce')
    state = st.selectbox('Select a State for Prediction', df['State_UT'].unique())
    education_level = st.selectbox('Select Education Level for Prediction', ['Primary', 'Upper Primary', 'Secondary', 'Higher Secondary'])
    state_data = df[df['State_UT'] == state]
    X = state_data[['year']].dropna()
    y = state_data[education_level].dropna()
    if len(X) != len(y):
        st.error('Insufficient data for prediction. Please try a different state or education level.')
    else:
        if st.button('Predict'):
            model = LinearRegression()
            model.fit(X, y)
            future_years = np.array([[year] for year in range(2016, 2021)]).reshape(-1, 1)
            predictions = model.predict(future_years)
            past_data = state_data[['year', education_level]].dropna()
            future_data = pd.DataFrame({'year': range(2016, 2021), education_level: predictions})
            plot_data = pd.concat([past_data, future_data])
            c1,c2 = st.columns(2)
            fig = px.line(plot_data, x='year', y=education_level, title=f'Predicted {education_level} Dropout Rates for {state}')
            fig.add_scatter(x=future_data['year'], y=future_data[education_level], mode='markers+lines', name='Predictions')
            c1.plotly_chart(fig)

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df_grouped = df.groupby("year")[num_cols].mean()
    merged = pd.merge(india, df, left_on="NAME_1", right_on="State_UT", how="inner")
    merged =merged[merged['NAME_1'] != 'Chandigarh']
    c1, c2 = st.columns(2)
    c1.subheader('Average Dropout Rates in India from 2012 to 2015')
    fig, ax = plt.subplots( figsize=(10, 9))
    ax.set_facecolor('#ffffff')
    merged.plot(column='Average', cmap='PuBu', linewidth=1, ax=ax, edgecolor='0.5', legend=True )
    for idx, row in merged.iterrows():
        centroid_x, centroid_y = row['geometry'].centroid.x, row['geometry'].centroid.y
        state_name = row['NAME_1']
        ax.text(centroid_x, centroid_y, state_name, fontsize=10, ha='center', va='center', color='black')
    c1.pyplot(fig, use_container_width=True)

    c2.subheader('Dropout Rates Correlation Matrix')
    df_total = df[['Primary','Upper Primary','Secondary','Higher Secondary']]
    correlation_matrix = df_total.corr()
    fig,ax=plt.subplots(figsize=(10,8.5))
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues')
    c2.pyplot()

    
elif analysis_type == "ENROLLMENT":

    rename_dict1={"Primary_Total": "Primary",
              "Upper_Primary_Total": "Upper Primary",
              "Secondary_Total": "Secondary",
              "Higher_Secondary_Total": "Higher Secondary"}
              
    @st.cache_data()
    def load_data():
        df = pd.read_csv('gross-enrollment-ratio-2013-2016.csv')
        df['Year'] = df['Year'].apply(lambda x: int(x.split('-')[0]))  # Convert Year to integer
        df.replace("NR", np.nan, inplace=True)
        df.rename(columns=rename_dict1, inplace=True)
        df["Primary"] = df["Primary"].astype(float)
        df["Upper Primary"] = df["Upper Primary"].astype(float)
        df["Secondary"] = df["Secondary"].astype(float)
        df["Higher Secondary"] = pd.to_numeric(df["Higher Secondary"], errors='coerce')
        df["Higher Secondary"].fillna(0, inplace=True)
        df["Higher Secondary"] = df["Higher Secondary"].astype(float)
        df_total = df[['Primary','Upper Primary','Secondary','Higher Secondary']]
        df["Average"] = df_total.mean(axis=1)


        

        return df
    

    with st.spinner("Processing Enrollment data...."):
        df = load_data()


    st.markdown("""<h2 style="text-align: center;">Enrollment Analysis and Visualization</h2>""", unsafe_allow_html=True)
    
    
    # Get unique states and education levels
    states = df["State_UT"].unique()
    education_levels = ['Primary', 'Upper Primary', 'Secondary', 'Higher Secondary']

    # Layout with two columns
    c1, c2 = st.columns(2)

    # User input in the left column
    c1.subheader("Select the State and Education Level")
    selected_state = c1.selectbox("State", states)
    selected_level = c1.selectbox("Education Level", education_levels)

    # Filter data based on selections
    df_state = df[df["State_UT"] == selected_state]

    # Display data in the left column
    c1.dataframe(df_state, use_container_width=True)

    # Visualization in the right column
    c2.subheader(f"Enrollment Rate for {selected_level} Level in {selected_state}")

    # Creating a bar plot
    fig = px.bar(
        data_frame=df_state,
        x="Year",
        y=f"{selected_level}",
        # title=f"Enrollment Rate for {selected_level} Level in {selected_state}",
        barmode="group",
        color_discrete_sequence=px.colors.diverging.Tealrose)

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Enrollment Rate (%)",
        margin=dict(t=50, b=50, l=50, r=50)
    )

    # Display the plot using Streamlit
    c2.plotly_chart(fig, use_container_width=True)


    # State-wise Comparison
    st.header(f"State-wise Enrollment Comparison")
    # st.sidebar.header("User Input Parameters")
    selected_year = st.selectbox("Select Year", df['Year'].unique(), index=0)
    selected_level1 = st.selectbox("Select Education Level", ['Primary', 'Upper Primary', 'Secondary', 'Higher Secondary'], index=0)
    state_wise_data = df[df['Year'] == selected_year]

    # Creating a bar plot with increased size
    fig = px.bar(state_wise_data, x="State_UT", y=f"{selected_level1}", 
                title=f"State-wise Enrollment in {selected_level1} Education ({selected_year})")

    # Update the layout to increase the size
    fig.update_layout(
        height=600,  # Height of the chart in pixels
        width=1200,   # Width of the chart in pixels
        xaxis_title="State/UT",
        yaxis_title="Enrollment Ratio",
        margin=dict(t=50, b=50, l=50, r=50),
        # polar=[ "#EF553B"],
    )

    # Display the plot using Streamlit
    st.plotly_chart(fig, use_container_width=False)  # Set to False to use the specified width


    c1,c2=st.columns(2)
    selected_state = c1.selectbox("Select State", df['State_UT'].unique(), index=0)
    selected_level = c1.selectbox("Select the Education Level", ['Primary', 'Upper Primary', 'Secondary', 'Higher Secondary'], index=0)

    # Gender Comparison
    c2.subheader(f"Gender Comparison in {selected_state} ({selected_year})")
    gender_data = df[(df['Year'] == selected_year) & (df['State_UT'] == selected_state)]
    fig = go.Figure(data=[go.Pie(labels=['Boys', 'Girls'], 
                                 values=[gender_data[f"{selected_level}_Boys"].values[0], gender_data[f"{selected_level}_Girls"].values[0]],
                                 marker_colors=px.colors.diverging.Temps,)])
    c2.plotly_chart(fig, use_container_width=True)


    
    if df['Year'].dtype == 'O':  # 'O' stands for object, typically used for strings in pandas
        df['Year'] = pd.to_numeric(df['Year'].str[:4], errors='coerce')

    # User selects a state and education level
    state = st.selectbox('Select a State for Prediction', df['State_UT'].unique())
    education_level = st.selectbox('Select Education Level for Prediction', ['Primary', 'Upper Primary', 'Secondary', 'Higher Secondary'])

    # Filter data based on the state
    state_data = df[df['State_UT'] == state]

    # Prepare data for model
    X = state_data[['Year']].dropna()
    y = state_data[education_level].dropna()

    # Check if data is sufficient for prediction
    if len(X) != len(y):
        st.error('Insufficient data for prediction. Please try a different state or education level.')
    else:
        # Predict future values when the button is clicked
        if st.button('Predict'):
            model = LinearRegression()
            model.fit(X, y)
            future_years = np.array([[year] for year in range(2017, 2022)]).reshape(-1, 1)
            predictions = model.predict(future_years)

            # Combine past data and future predictions
            past_data = state_data[['Year', education_level]].dropna()
            future_data = pd.DataFrame({'Year': range(2017, 2022), education_level: predictions})
            plot_data = pd.concat([past_data, future_data])

            # Plotting
            c1, c2 = st.columns(2)
            fig = px.line(plot_data, x='Year', y=education_level, title=f'Predicted {education_level} Enrollment Rates for {state}')
            fig.add_scatter(x=future_data['Year'], y=future_data[education_level], mode='markers+lines', name='Predictions')
            c1.plotly_chart(fig, use_container_width=True)


        regions = {
        "North India": [
            'Haryana', 'Himachal Pradesh', 'Jammu And Kashmir', 'Punjab', 'Uttar Pradesh', 
             'Delhi', 'Chandigarh'
        ],
        "South India": [
            'Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu', 'Telangana', 
            'Puducherry', 'Lakshadweep'
        ],
        "East India": [
            'Bihar', 'Jharkhand', 'Odisha', 'West Bengal', 'Andaman & Nicobar Islands'
        ],
        "West India": [
            'Goa', 'Gujarat', 'Maharashtra', 'Rajasthan', 'Dadra & Nagar Haveli', 
            'Daman & Diu'
        ],
        "Central India": [
            'Chhattisgarh', 'Madhya Pradesh', 'MADHYA PRADESH'
        ],
        "North East India": [
            'Arunachal Pradesh', 'Assam', 'Manipur', 'Meghalaya', 'Mizoram', 
            'Nagaland', 'Sikkim', 'Tripura'
        ],
        "Union Territories": [
            'Pondicherry', 'Delhi', 'Chandigarh', 'Dadra & Nagar Haveli', 'Daman & Diu', 
            'Lakshadweep', 'Andaman & Nicobar Islands', 'Puducherry'
        ],
    
    }

    # Streamlit UI
    st.header('Region wise Enrollment Data Visualization')

    # Dropdown for selecting the region
    selected_region = st.selectbox("Select Region", list(regions.keys()))

    # Dropdown for selecting the education level
    education_levelss = ['Primary', 'Upper Primary', 'Secondary', 'Higher Secondary'," "]  # Modify as per your column names
    selected_level = st.selectbox("Select Education Level", education_levelss)

    # Filter data based on the selected region and education level
    filtered_df = df[df['State_UT'].isin(regions[selected_region])]
    filtered_df = filtered_df[filtered_df['Year'].isin([2013, 2014, 2015])]

    # Pivot the DataFrame
    pivot_df = filtered_df.pivot(index="State_UT", columns="Year", values=selected_level).reset_index()

    # Prepare data for Plotly
    plotly_data = pivot_df.melt(id_vars=['State_UT'], var_name='Year', value_name=selected_level)

    # Create the heatmap with Plotly
    fig = px.imshow(plotly_data.pivot(index='State_UT', columns='Year', values=selected_level),
                    labels=dict(x="Year", y="State/UT", color=selected_level),
                    x=plotly_data['Year'].unique(),
                    y=plotly_data['State_UT'].unique(),
                    title=f'Enrollment Rate Heatmap for {selected_level} in {selected_region}',
                    color_continuous_scale='Viridis'
                    )

    # Update the layout
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[2013, 2014, 2015],
            ticktext=['2013', '2014', '2015']
        )
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "STATES WITH FACILITIES":
    
    # Load Data
    @st.cache_data
    def load_data():
        elec = pd.read_csv('percentage-of-schools-with-electricity-2013-2016.csv')
        # data = pd.read_csv('percentage-of-schools-with-comps-2013-2016.csv')
        return elec

    electricity= load_data()
    def load_computer_data():
        comps = pd.read_csv('percentage-of-schools-with-comps-2013-2016.csv')
        return comps

    computers = load_computer_data()

    # Display Data
    if st.checkbox('Show Raw Data'):
        c1,c2 = st.columns(2)
        c1.subheader('Electricity Data')
        c2.subheader('Computer Data')
        c1.write(electricity)
        c2.write(computers)
        # c2.write(data)

        # Display Data Option for Computers
    # if st.checkbox('Show Computer Data'):
    #     st.subheader('Computer Data')
    #     c1, c2 = st.columns(2)
    #     c2.write(computers)  # Assuming c2 is where you want to display the computer data

    c1,c2=st.columns(2)
    percentage_with_electricity = electricity['All Schools'].mean()

    # Calculate the percentage of schools without electricity
    percentage_without_electricity = 100 - percentage_with_electricity

    # Data for pie chart
    pie_data = {
        'Category': ['With Electricity', 'Without Electricity'],
        'Percentage': [percentage_with_electricity, percentage_without_electricity]
    }

    df_pie = pd.DataFrame(pie_data)

    # Create the pie chart
    fig = px.pie(df_pie, names='Category', values='Percentage', title='Electricity Availability in Schools')

    # Display the pie chart
    c1.plotly_chart(fig, use_container_width=True)

    # Analysis for Computers (similar to electricity)
    # c1, c2 = st.columns(2)  # You may adjust the columns depending on how you want to layout the visualization

    # Calculate average percentage of schools with computers
    percentage_with_computers = computers['All Schools'].mean()

    # Calculate the percentage of schools without computers
    percentage_without_computers = 100 - percentage_with_computers

    # Data for computer availability pie chart
    pie_data_computers = {
        'Category': [ 'Without Computers','With Computers'],
        'Percentage': [percentage_with_computers, percentage_without_computers]
    }

    df_pie_computers = pd.DataFrame(pie_data_computers)

    # Create the pie chart for computer availability
    fig_computers = px.pie(df_pie_computers, names='Category', values='Percentage', 
                        title='Computer Availability in Schools')

    # Display the pie chart for computer availability
    c2.plotly_chart(fig_computers, use_container_width=True)

    #################

    yearly_electricity = electricity.groupby('year')['All Schools'].mean().reset_index()
    yearly_computer = computers.groupby('year')['All Schools'].mean().reset_index()

    # Create the line chart
    fig1 = px.line(yearly_electricity, x='year', y='All Schools', 
                title='Rate of Electricity Availability in Schools (2013-2016)',
                labels={'year': 'Year', 'All Schools': 'Percentage with Electricity'})
    
    fig2 = px.line(yearly_computer, x='year', y='All Schools',
                title='Rate of Computer Availability in Schools (2013-2016)',
                labels={'year': 'Year', 'All Schools': 'Percentage with Computers'})
    
    c1, c2 = st.columns(2)

    # Display the line chart
    c1.plotly_chart(fig1, use_container_width=True)
    c2.plotly_chart(fig2, use_container_width=True)



    regions = {
        "North India": [
            'Haryana', 'Himachal Pradesh', 'Jammu And Kashmir', 'Punjab', 'Uttar Pradesh', 
             'Delhi', 'Chandigarh'
        ],
        "South India": [
            'Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu', 'Telangana', 
            'Puducherry', 'Lakshadweep'
        ],
        "East India": [
            'Bihar', 'Jharkhand', 'Odisha', 'West Bengal', 'Andaman & Nicobar Islands'
        ],
        "West India": [
            'Goa', 'Gujarat', 'Maharashtra', 'Rajasthan', 'Dadra & Nagar Haveli', 
            'Daman & Diu'
        ],
        "Central India": [
            'Chhattisgarh', 'Madhya Pradesh', 'MADHYA PRADESH'
        ],
        "North East India": [
            'Arunachal Pradesh', 'Assam', 'Manipur', 'Meghalaya', 'Mizoram', 
            'Nagaland', 'Sikkim', 'Tripura'
        ],
        "Union Territories": [
            'Pondicherry', 'Delhi', 'Chandigarh', 'Dadra & Nagar Haveli', 'Daman & Diu', 
            'Lakshadweep', 'Andaman & Nicobar Islands', 'Puducherry'
        ],
    
    }

    c1, c2 = st.columns(2)

    # Create a state to region mapping
    state_to_region = {state: region for region, states in regions.items() for state in states}

    # Map each state to its region
    electricity['Region'] = electricity['State_UT'].map(state_to_region)

    # Order the regions in the DataFrame as per your defined order
    ordered_regions = list(regions.keys())
    electricity['Region'] = pd.Categorical(electricity['Region'], categories=ordered_regions, ordered=True)

    # Group by region
    region_wise_electricity = electricity.groupby('Region')['All Schools'].mean().reset_index()

    # Create the funnel chart
    fig = px.funnel(data_frame=region_wise_electricity, x='Region', y='All Schools',
                    labels={'Region': 'Region', 'All Schools': 'Average Percentage'},
                    color_discrete_sequence=['teal'], height=500)

    # Display the chart
    c1.header("Region Wise Average Percentage of Schools with Electricity")
    c1.plotly_chart(fig, use_container_width=True)
    ######################3


    

    

    computers['Region'] = computers['State_UT'].map(state_to_region)

    # Order the regions in the DataFrame as per your defined order for the computer data
    computers['Region'] = pd.Categorical(computers['Region'], categories=ordered_regions, ordered=True)

    # Group by region for the computer data
    region_wise_computers = computers.groupby('Region')['All Schools'].mean().reset_index()

    # Create the funnel chart for the computer data
    fig_computers = px.funnel(data_frame=region_wise_computers, x='Region', y='All Schools',
                            labels={'Region': 'Region', 'All Schools': 'Average Percentage'},
                            color_discrete_sequence=['teal'], height=500)

    # Display the funnel chart for the computer data
    c2.header("Region Wise Average Percentage of Schools with Computers")
    c2.plotly_chart(fig_computers, use_container_width=True)


################
    states = electricity['State_UT'].unique().tolist()  # Simplified, assuming all states are in both datasets

    # Dropdown for state selection
    selected_state = st.selectbox('Select a State', states)

    # Filter data based on selected state
    electricity_state = electricity[electricity['State_UT'] == selected_state]
    computers_state = computers[computers['State_UT'] == selected_state]
    c1, c2 = st.columns(2)
    # Example visualization: Percentage of schools with electricity in the selected state over the years
    if not electricity_state.empty:
        fig_electricity = px.line(electricity_state, x='year', y='All Schools',
                                title=f'Electricity Availability in Schools - {selected_state}')
        c1.plotly_chart(fig_electricity, use_container_width=True)

    # Example visualization: Percentage of schools with computers in the selected state over the years
    if not computers_state.empty:
        fig_computers = px.line(computers_state, x='year', y='All Schools',
                                title=f'Computer Availability in Schools - {selected_state}')
        c2.plotly_chart(fig_computers, use_container_width=True)

    









