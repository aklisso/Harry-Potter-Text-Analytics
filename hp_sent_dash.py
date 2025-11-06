from dash import Dash, html, dcc, Input, Output
import os
import pickle
from collections import Counter
from wordcloud import WordCloud
import pandas as pd
import polars as pl
import polars.selectors as cs
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import (
    Dash,
    html,
    dcc,
    Input,
    Output,
)
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import base64
import matplotlib as mpl
import matplotlib.pyplot as plt

# [Load stylesheets and initialize app]
# DCC_BOOTSTRAP_CSS = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.8/dbc.min.css"
load_figure_template('BOOTSTRAP')
ext_ss = [
    dbc.themes.BOOTSTRAP,
    # DCC_BOOTSTRAP_CSS
]
app = Dash(__name__,
           external_stylesheets=ext_ss)
app.title = 'Harry Potter Sentiment Analysis'

# [Load in data]

# TODO: change file name as needed
SENTIMENT_DATA_FILE = 'sentences_sentiment_minmax.csv'
sent = pl.read_csv(SENTIMENT_DATA_FILE)

# modifying data
titles = {
    1: '1 - Sorcerers Stone',
    2: '2 - Chamber of Secrets',
    3: '3 - Prisoner of Azkaban',
    4: '4 - Goblet of Fire',
    5: '5 - Order of the Phoenix',
    6: '6 - Half-Blood Prince',
    7: '7 - Deathly Hallows'
}
chapter_sent = (
    sent
    .with_columns(
        book_num=pl.col('book_num'),
        # sentence index partitioned by book
        book_sent_index=pl.arange(0, pl.len()).over('book_num'),
    )
    .with_columns(
        # percentile of all sentences in the book before this sentence
        book_progress=pl.col('book_sent_index') /
        pl.col('book_sent_index').max().over('book_num')
    )
    .group_by('book_num', 'chapter_num')
    .agg(
        valence=pl.col('valence').mean(),
        valence_std=pl.col('valence').std(),
        arousal=pl.col('arousal').mean(),
        arousal_std=pl.col('arousal').std(),
        book_progress=pl.col('book_progress').mean()
    )
    .with_columns(
        valence_scaled=(pl.col('valence') - pl.col('valence').min()) /
        (pl.col('valence').max() - pl.col('valence').min()),
        arousal_scaled=(pl.col('arousal') - pl.col('arousal').min()) /
        (pl.col('arousal').max() - pl.col('arousal').min()),
    )
    .with_columns(
        index_string=pl.concat_str(pl.lit('Book '),
                                   pl.col('book_num'),
                                   pl.lit(', Chapter '),
                                   pl.col('chapter_num')
                                   ),
        series_ch_num=pl.arange(
            1, pl.len()+1).over(order_by=['book_num', 'chapter_num'])
    )
    .join(
        pl.DataFrame({'book_num': titles.keys(),
                     'title': titles.values()})
        .with_columns(just_title=pl.col('title').str.replace(r'\d\s*-\s*', '')),
        on='book_num'
    )
    .sort('book_num', 'chapter_num')
)

sent_pd = sent.to_pandas()
chapter_sent_pd = chapter_sent.to_pandas()

# scaling for graph
pad_factor = 0.05
valence_scaled_min = chapter_sent['valence_scaled'].min()
valence_scaled_max = chapter_sent['valence_scaled'].max()
valence_pad = 0.05 * (valence_scaled_max - valence_scaled_min)
arousal_scaled_min = chapter_sent['arousal_scaled'].min()
arousal_scaled_max = chapter_sent['arousal_scaled'].max()
arousal_pad = 0.05 * (arousal_scaled_max - arousal_scaled_min)

# [Creating Dash components]
book_nums = list(titles.keys())

filter_component = dcc.Dropdown(
    id='filter-dropdown',
    multi=True,
    options=[
        {'label': titles[i], 'value': i}
        for i in book_nums
    ],
    value=book_nums,
)


@app.callback(
    Output('filter-dropdown', 'value'),
    Input('filter-dropdown', 'value')
)
def update_dropdown_sorted(selected: list):
    if selected is not None:
        return sorted(selected)
    return selected


checklist_box = dbc.Card(
    children=[
        dbc.CardHeader('Books'),
        dbc.CardBody([html.Div(filter_component, className='dbc')])
    ],
    className=""
)

# create an empty dcc.Graph as a placeholder.
# when the dashboard is created it will use the callback-
# decorated function (`update_circumplex`) to initialize the graph
circumplex_graph_component = dcc.Graph(
    id='circumplex-graph'
)


@app.callback(
    Output('circumplex-graph', 'figure'),
    Input('filter-dropdown', 'value')
)
def update_circumplex(selected: list):
    '''Create the plotly figure for the circumplex.'''
    df_filter = chapter_sent.filter(pl.col('book_num').is_in(selected))
    fig = px.scatter(
        df_filter,
        x='valence_scaled',
        y='arousal_scaled',
        color='book_progress',
        custom_data=['book_progress', 'just_title', 'chapter_num']
    )

    fig.update_xaxes(
        title_text='Valence (Scaled)',
        range=[valence_scaled_min - valence_pad,
               valence_scaled_max + valence_pad],
    )
    fig.update_yaxes(
        title_text='Arousal (Scaled)',
        range=[arousal_scaled_min - arousal_pad,
               arousal_scaled_max + arousal_pad],
    )
    fig.update_traces(
        selector=dict(type='scatter'),
        marker=dict(
            cmin=0,
            cmax=1
        ),
        hovertemplate=(
            '<em>%{customdata[1]}</em>, <b>Chapter %{customdata[2]}</b><br>'
            'Valence (Scaled): %{x:.4f}<br>'
            'Arousal (Scaled): %{y:.4f}<br>'
            'Book Progress: %{customdata[0]:.0%}'
            '<extra></extra>'
        )
    )
    fig.update_coloraxes(
        colorbar=dict(
            title='Book Progress',
            tickmode='array',
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1]
        )
    )
    fig.update_layout(
        title=dict(text='Valence and Arousal by Book Progress',
                   font=dict(size=24, weight='bold'),
                   x=0.5,
                   xanchor='center')
    )

    return fig
############################################# Hannah Line Graph #############################################

# Define title names for line graph


def Titles(x):
    """Maps book number to title."""
    titles_map = {
        1: "Sorcerer's Stone",
        2: "Chamber of Secrets",
        3: "Prisoner of Azkaban",
        4: "Goblet of Fire",
        5: "Order of the Phoenix",
        6: "Half Blood Prince",
        7: "Deathly Hallows"
    }
    return titles_map.get(x, f"Book {x}")


# Create data used in line graph
line_graphs_data = (
    sent.with_columns(
        pl.col('book_num').cast(pl.Int64, strict=False),
        pl.col('chapter_num').cast(pl.Int64, strict=False),
        pl.col('compound').cast(pl.Float64, strict=False),)
    # Map book_num to BookTitle
    .with_columns(pl.col('book_num').map_elements(Titles, return_dtype=pl.String).alias('BookTitle'))
    # Select only the required columns and filter out NaNs
    .select(['chapter_num', 'BookTitle', 'compound'])
    .filter(pl.col('compound').is_not_null())
    # Calculate mean compound sentiment per chapter and book
    .group_by(['chapter_num', 'BookTitle'])
    .agg(pl.col('compound').mean().alias('compound'))
    .sort(['BookTitle', 'chapter_num']))

# Define colors and yaxis limits
colors_list = ["#a38265", "#aa2550", "#cc7b75",
               "#5f9372", "#1573a4", "#3f5a3a", "#f1893f"]
# Define the order of the books (Titles)
book_order = [Titles(i) for i in range(1, 8)]
color_discrete_map = dict(zip(book_order, colors_list)
                          )  # Map book titles to colors
y_limit = [-0.25, 0.25]

# Create an initial figure and the component
initial_line_graph = go.Figure()
initial_line_graph.update_layout(
    title={'text': "Select a Book Title to View Sentiment Trend",
           'x': 0.5, 'xanchor': 'center'},
    height=400)

books_line_graph_component = dcc.Graph(
    id='books_line_graph',
    figure=initial_line_graph)

# Map book numbers to names


def get_titles_from_nums(book_nums_list):
    """Converts a list of book numbers to a list of full book titles."""
    return [Titles(num) for num in book_nums_list]


@app.callback(
    Output('books_line_graph', 'figure'),
    [Input('filter-dropdown', 'value')])
def update_graph(selected_book_nums):
    if not selected_book_nums:  # Return placeholder if nothing is selected
        return go.Figure().update_layout(
            title={'text': "Select one or more Book Titles.",
                   'x': 0.5, 'xanchor': 'center'},
            height=400)

    # The dropdown returns book numbers (integers) --> convert them to the full titles (strings)
    selected_titles = get_titles_from_nums(selected_book_nums)
    num_books = len(selected_titles)

    # Create Subplots dynamically based on the number of selected books
    books_line_graph = make_subplots(
        rows=num_books, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,)

    # Loop through each selected book to add its trace and custom annotations
    for i, title in enumerate(selected_titles):
        df_book = line_graphs_data.filter(
            pl.col('BookTitle') == title).to_pandas()
        # Get the corresponding color
        color = color_discrete_map.get(title, '#000000')
        books_line_graph.add_trace(
            go.Scatter(
                x=df_book['chapter_num'],
                y=df_book['compound'],
                mode='lines',
                name=title,
                line=dict(color=color, width=2),
                showlegend=False),
            row=i + 1, col=1)

        # Add Book Name Annotation
        x_ref_name = 'x' if i == 0 else f'x{i + 1}'
        y_ref_name = 'y' if i == 0 else f'y{i + 1}'
        books_line_graph.add_annotation(
            text=f"<b>{title}</b>", xref=f"{x_ref_name} domain", yref=f"{y_ref_name} domain", x=1.0, y=1.0,
            showarrow=False, font=dict(size=12, color=color), xanchor='right', yanchor='top')

        # Axis Customizations
        books_line_graph.update_yaxes(
            range=y_limit,
            title_text="Compound Sentiment Score" if i == num_books // 2 else None,
            row=i + 1,
            col=1,
            tickfont=dict(size=12))
        if i < num_books - 1:  # Hide X-axis tick labels for all but the bottom plot
            books_line_graph.update_xaxes(
                showticklabels=False, row=i + 1, col=1)

    books_line_graph.update_traces(
        hovertemplate=(
            "<b>Chapter %{x}</b><br>" +
            # Formats to 4 decimal places
            "Compound Sentiment: %{y:.4f}<extra></extra>"
        ))

    # Title Customizations
    books_line_graph.update_layout(
        title={
            'text': "Compound Sentiment Scores Over Time",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'weight': 'bold'}},
        xaxis_title=None, **{f'xaxis{num_books}': {'title': 'Chapter'}},
        height=520,
        margin=dict(l=50, r=50, t=80, b=50))

    return books_line_graph


############################################### End of Hannah Line Graph ##############################################

############################################### Ava Line Graph ##############################################
# chapter_sentiment_minmax = pd.read_csv("chapter_sentiment_minmax.csv")
# using ch_sent instead

# Create an empty figure with subplots to act as a placeholder
initial_parallel_fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True  # ,
    # subplot_titles=('Valence over time', 'Arousal over time') Removing titles for cleaner layout
)

initial_parallel_fig.update_layout(
    title={'text': "Select one or more books to view sentiment.",
           'x': 0.5, 'xanchor': 'center'},
    height=450
)

# Add axis titles
initial_parallel_fig.update_yaxes(
    title_text="Valence (Percentile)", row=1, col=1)
initial_parallel_fig.update_yaxes(
    title_text="Arousal (Percentile)", row=2, col=1)
initial_parallel_fig.update_xaxes(title_text="Chapter number", row=2, col=1)

# --- THIS IS YOUR NEW, COMBINED CALLBACK ---


@app.callback(
    Output('parallel-line-plots', 'figure'),  # <-- Single output
    Input('filter-dropdown', 'value')
)
def update_sentiment_graphs(selected_book_nums):

    # 1. Handle the case where no books are selected
    if not selected_book_nums:
        # Return the empty placeholder figure
        return initial_parallel_fig

    # 2. Filter your main dataframe (same as before) - note: I removed filter b/c it was deleting parts of my graphs
    filtered_df = chapter_sent_pd  # [
    #     chapter_sent_pd['book_num'].isin(selected_book_nums)
    # ]
    # Filtering strategy:
    # - We need to index

    #
    # --- 3. Create Valence & Arousal figs (same as before) ---
    # (We use these as temporary "trace generators")
    #

    # --- VALENCE ---
    valence_graph = px.line(filtered_df, x="series_ch_num", y="valence_scaled", color='book_num',
                            color_discrete_map={
                                1: "#a38265",
                                2: "#aa2550",
                                3: "#cc7b75",
                                4: "#5f9372",
                                5: "#1573a4",
                                6: "#3f5a3a",
                                7: "#f1893f"
                            })
    valence_graph.for_each_trace(lambda trace: trace.update(name=trace.name
                                                            .replace("1", "Sorcerer's Stone")
                                                            .replace("2", "Chamber of Secrets")
                                                            .replace("3", "Prisoner of Azkaban")
                                                            .replace("4", "Goblet of Fire")
                                                            .replace("5", "Order of the Phoenix")
                                                            .replace("6", "Half-Blood Prince")
                                                            .replace("7", "Deathly Hallows")))
    valence_graph.update_traces(hovertemplate=None, hoverinfo="skip")
    # (Your static peak data and add_trace for valence)
    val_peak_x_coords = [64, 91, 110, 130, 160, 165, 194]
    val_peak_y_coords = [1, 0, 0.885, 0.238, 0.189, 0.868, 0.255]
    val_peak_hover_text = [
        "Harry attends the <br>Quidditch World cup <br>with Hermione and <br>the Weasley family.",
        "Voldemort challenges Harry to <br>a duel. Harry manages to repel <br>Voldemort's 'Avada Kedavra' <br>curse.",
        "Harry agrees to <br>lead a student-run Defense <br>Against the Dark Arts <br>group. Many students sign up <br>to attend.",
        "Harry tries to avenge <br>Sirius' murder. Voldemort <br>enters Harry's body.",
        "Harry battles with Snape <br>and other Death Eaters. <br>Dumbledore dies, and <br>Harry finds his body.",
        "The Dursleys go into <br>hiding. Before Dudley leaves, <br>he reveals he is grateful <br>that Harry saved his <br>life the previous <br>summer.",
        "Harry, Ron, and Hermione <br>fight off Death Eaters <br>and Dementors. Voldemort <br>kills Snape."
    ]
    valence_graph.add_trace(go.Scatter(
        x=val_peak_x_coords, y=val_peak_y_coords, mode='markers',
        marker=dict(color='#494235', size=8, symbol="star-diamond"),
        hovertext=val_peak_hover_text, hovertemplate='%{hovertext}<extra></extra>', showlegend=False
    ))

    # --- AROUSAL ---
    arousal_graph = px.line(filtered_df, x="series_ch_num", y="arousal_scaled", color='book_num',
                            color_discrete_map={
                                1: "#a38265",
                                2: "#aa2550",
                                3: "#cc7b75",
                                4: "#5f9372",
                                5: "#1573a4",
                                6: "#3f5a3a",
                                7: "#f1893f"
                            })
    arousal_graph.for_each_trace(lambda trace: trace.update(name=trace.name
                                                            .replace("1", "Sorcerer's Stone")
                                                            .replace("2", "Chamber of Secrets")
                                                            .replace("3", "Prisoner of Azkaban")
                                                            .replace("4", "Goblet of Fire")
                                                            .replace("5", "Order of the Phoenix")
                                                            .replace("6", "Half-Blood Prince")
                                                            .replace("7", "Deathly Hallows")))
    arousal_graph.update_traces(hovertemplate=None, hoverinfo="skip")
    # (Your static peak data and add_trace for arousal)
    ar_peak_x_coords = [18, 37, 63, 91, 101, 130, 163, 198]
    ar_peak_y_coords = [0.4639, .08, 0, 1, 0.0499, 0.57, 0.667, .609]
    ar_peak_hover_text = [
        "Dumbledore saves Harry from <br>Professor Quirrel, who reveals <br>himself to be working for <br>Voldemort",
        "Harry's aunt visits <br>and he tries to be <br>polite, but she criticizes his <br>late parents. He casts a <br>spell on her to <br>make her float away.",
        "Harry experiences a peaceful <br>visit to the Weasleys' <br>home.",
        "Voldemort challenges Harry to <br>a duel. Harry manages to repel <br>Voldemort's 'Avada Kedavra' <br>curse.",
        "Harry travels to his <br>Ministry of Magic hearing <br>with Mr. Weasley.",
        "Harry tries to avenge <br>Sirius' murder. Voldemort <br>enters Harry's body.",
        "Voldemort conspires with <br>Death Eaters. He murders <br>a Hogwards professor, whom <br>he held captive.",
        "Harry kills Voldemort. <br>He speaks to Dumbledore's <br>portrait and reveals his <br>intentions to return the <br>Elder Wand to Dumbledore's <br>grave."
    ]
    arousal_graph.add_trace(go.Scatter(
        x=ar_peak_x_coords, y=ar_peak_y_coords, mode='markers',
        marker=dict(color='#494235', size=8, symbol="star-diamond"),
        hovertext=ar_peak_hover_text, hovertemplate='%{hovertext}<extra></extra>', showlegend=False
    ))

    #
    # --- 4. Create the new subplot figure and copy traces ---
    #
    parallel_line_plots = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True  # ,
        # subplot_titles=('Valence over time', 'Arousal over time') # Subplot titles
    )

    # Copy all traces from the valence graph to the 1st row
    for trace in valence_graph.data:
        parallel_line_plots.add_trace(trace, row=1, col=1)

    # Copy all traces from the arousal graph to the 2nd row
    for trace in arousal_graph.data:
        # Hide the legend for the 2nd plot to avoid duplicates
        if trace.mode == 'lines':
            trace.showlegend = False
        parallel_line_plots.add_trace(trace, row=2, col=1)

    # --- 5. Update layout for the combined figure ---
    parallel_line_plots.update_layout(
        title={'text': f"Valence and Arousal Over Time",
               'font': {'size': 24, 'weight': 'bold'},
               'x': 0.5,
               'xanchor': 'center'},
        height=450,
        legend_title_text='Book',
        xaxis_title=None,  # Hide top x-axis title
        xaxis2_title='Total Chapters Elapsed',  # Show bottom x-axis title
        yaxis_title='Valence (Scaled)',
        yaxis2_title='Arousal (Scaled)'
    )

    # 6. Return the single, combined figure
    return parallel_line_plots


################################################# End of Ava Line Graph ###############################################
############################################### Jamil's Wordcloud ##############################################

with open('character_raw_counts.pkl', 'rb') as fp:
    character_freqs = pickle.load(fp)

wordcloud_img_component = html.Img(id='wordcloud-img-component',
                                   className='img-fluid')

wordcloud_component = dbc.Card(
    dbc.CardBody(
        children=[
            html.H3('Characters Wordcloud',
                    style={
                        'textAlign': 'center',
                        'fontWeight': 'bold',
                        'fontSize': '24px'
                    }),
            wordcloud_img_component
        ]
    ),
    className='border-0'
)


@app.callback(
    Output('wordcloud-img-component', 'src'),
    Input('filter-dropdown', 'value')
)
def update_wordcloud(selected: list):
    selected_counts = Counter()
    for s in selected:
        selected_counts.update(character_freqs[s])
    selected_counts = Counter(dict(selected_counts.most_common(50)))

    # min_count = min(selected_counts.values())
    max_count = max(selected_counts.values())
    selected_counts = Counter({
        w: c / max_count
        for w, c in selected_counts.items()
    })

    selected_ranks = {
        w: r
        for r, (w, c) in enumerate(selected_counts.most_common())
    }
    total_words = len(selected_ranks) - 1

    COLORMAP_NAME = 'plasma'
    colormap = plt.get_cmap(COLORMAP_NAME)

    def frequency_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        rank = selected_ranks[word]
        # print(f'{rank=}, {total_words=}')
        rgba = colormap(rank / (total_words * 1.5))
        r, g, b, _ = [int(c * 255) for c in rgba]
        return f'rgb({r}, {g}, {b})'

    # code to create word cloud based on character_counts
    wordcloud = WordCloud(
        width=1000,
        height=1000,
        background_color="white",
        color_func=frequency_color_func,
        font_path='./assets/Lora-VariableFont_wght.ttf',
        prefer_horizontal=1.0,
    ).generate_from_frequencies(selected_counts)

    # code to convert the wordcloud to an image in memory
    img = wordcloud.to_image()
    with BytesIO() as buffer:
        img.save(buffer, format='PNG')

        # 4. Encode the binary data to a Base64 string
        img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    base64_uri = f'data:image/png;base64,{img_data}'

    # return the uri to the img src
    return base64_uri


# TODO: add new components to layout
app.layout = dbc.Container(
    className='dbc h-100',
    children=[
        dbc.Row(
            children=[
                dbc.Col(html.H1('Harry Potter Sentiment Analysis'),
                        className='d-flex align-items-center'),
                dbc.Col(filter_component,
                        className='align-items-center')
            ],
            style={'height': '10%'}
        ),
        dbc.Row(
            children=[
                dbc.Col(books_line_graph_component,
                        width=7),
                dbc.Col(wordcloud_component,
                        width=5)
            ],
            style={'height': '45%'}
        ),
        dbc.Row(
            children=[
                dbc.Col(circumplex_graph_component,
                        width=5),
                dbc.Col(children=[dcc.Graph(id='parallel-line-plots',
                                            figure=initial_parallel_fig)],
                        width=7)
            ],
            style={'height': '45%'}
        )
    ]
)

DEBUG = False
if __name__ == '__main__':
    app.run(debug=DEBUG)
