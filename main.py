from audioop import cross
from email import header
from email.policy import default
import json
from pathlib import Path

import arrow
import click
import dash
from dash import dcc, html, Input, Output
from flask import request
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import plotly.express as px

import bbceas_processing


@click.group()
def cli():
    pass


@cli.command()
@click.argument("in_data", type=click.File())
@click.option("-c", "--cross_sections_in", type=click.File(), multiple=True)
# @click.argument("cross_sections_2", type=click.File())
@click.argument("out_folder", type=click.Path(dir_okay=True, file_okay=False))
@click.option(
    "-i",
    "--instrument_type",
    type=click.Choice(["open-cavity", "closed-cavity"], case_sensitive=False),
    default="closed-cavity",
)
@click.option("-b", "--bounds_file", type=click.File())
def analyze(in_data, cross_sections_in, out_folder, instrument_type, bounds_file):
    # Load data
    in_data = pd.read_pickle(in_data.name)

    # Load in cross sections
    cross_sections = []
    for file in cross_sections_in:
        cross_sections.append(pd.read_csv(file, header=None, index_col=0))

    # Take the wavelengths from the cross-sections before sending to bounds picker
    in_data.columns = cross_sections[0].index

    if bounds_file is None:
        # Pick a specific wavelength for the bounds picker to display
        wavelengths = in_data.columns
        selected_wavelength = wavelengths[(wavelengths > 308) & (wavelengths < 312)][0]
        bounds = run_bounds_picker(in_data[selected_wavelength], instrument_type)
        print(bounds)
    else:
        bounds = json.load(bounds_file)

    if instrument_type == "closed-cavity":
        instrument = bbceas_processing.closed_cavity_data.ClosedCavityData()
    elif instrument_type == "open-cavity":
        instrument = bbceas_processing.open_cavity_data.OpenCavityData()

    processed_data = bbceas_processing.analyze(
        in_data, bounds, cross_sections, instrument
    )
    print(processed_data)

    save_data(processed_data, out_folder)


@cli.command(name="import")
@click.argument(
    "in_folder", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument("out_data", type=click.Path(file_okay=True, dir_okay=False))
@click.option(
    "--format", type=click.Choice(["asc"], case_sensitive=False), default="asc"
)
def import_data(in_folder, out_data, format):
    if format == "asc":
        data = bbceas_processing.utils.process_asc(in_folder)
    else:
        print(f"Unknown format: {format}")
        exit(1)

    data = data.sort_index()
    data.to_pickle(out_data)


def save_data(processed_data, out_folder):
    out_folder = Path(out_folder)

    # Save cross section plot
    cross_sections_target = processed_data["cross_sections_target"]
    plt.plot(cross_sections_target.index, cross_sections_target)
    plt.savefig(out_folder / "cross_sections_target.png")
    plt.cla()

    # returns the timestamp of associated with the highest concentration
    index_max_conc = processed_data["fit_curve_values"].idxmax()[0]

    fitted_data = processed_data["fit_data"].loc[[index_max_conc]].squeeze()
    absorption = processed_data["absorption"].loc[[index_max_conc]].squeeze()
    residuals = processed_data["residuals"].loc[[index_max_conc]].squeeze()

    for i in range(len(processed_data["fit_curve_values"].columns) - 3):
        if i == 0:
            title = "concentrations_target.png"
        else:
            title = "concentrations_" + str(i) + ".png"
        concentration = processed_data["fit_curve_values"][i]
        plt.plot(concentration.index, concentration)
        plt.savefig(out_folder / title)
        plt.cla()

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 1, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1:, :])

    ax1.plot(residuals.index, residuals, ".-")
    ax2.plot(fitted_data.index, fitted_data)
    ax2.plot(absorption.index, absorption)
    plt.savefig(out_folder / "results.png")
    plt.cla()


def run_bounds_picker(data, instrument_type):
    from collections import defaultdict

    bounds = defaultdict(lambda: [None, None])
    app = dash.Dash(__name__)

    fig = px.line(data)
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Intensity")
    fig.update_layout(dragmode="select", hovermode=False)

    if instrument_type == "closed-cavity":
        radio_items = dcc.RadioItems(
            id="radio-select",
            options=[
                {"label": "Dark Count   ", "value": "darkcounts"},
                {"label": "N2   ", "value": "N2"},
                {"label": "He   ", "value": "He"},
                {"label": "Target Sample", "value": "Target"},
            ],
        )

    if instrument_type == "open-cavity":
        radio_items = dcc.RadioItems(
            id="radio-select",
            options=[
                {"label": "Dark Count   ", "value": "darkcounts"},
                {"label": "Calibration   ", "value": "Calibration"},
                {"label": "Target Sample", "value": "Target"},
            ],
        )
    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Graph(id="one-wavelength", figure=fig),
            html.Br(),
            radio_items,
            html.Div(id="placeholder"),
            html.Br(),
            html.Link("Analyze Data", href="/analyze"),
            html.Br(),
            dcc.Link("Done", href="/shutdown"),
            html.Div(id="page-content"),
        ]
    )

    @app.callback(
        Output("placeholder", "children"),
        Input("radio-select", "value"),
        Input("one-wavelength", "selectedData"),
    )
    def getSelection(radio_select, graph_select):
        nonlocal bounds
        nonlocal instrument_type
        ran = graph_select["range"]["x"]

        if radio_select == "darkcounts":
            bounds["dark"][0] = ran[0]
            bounds["dark"][1] = ran[1]

        if radio_select == "Target":
            bounds["target"][0] = ran[0]
            bounds["target"][1] = ran[1]

        if instrument_type == "open-cavity":
            if radio_select == "Calibration":
                bounds["calibration"][0] = ran[0]
                bounds["calibration"][1] = ran[1]

        if instrument_type == "closed-cavity":
            if radio_select == "N2":
                bounds["N2"][0] = ran[0]
                bounds["N2"][1] = ran[1]
            if radio_select == "He":
                bounds["He"][0] = ran[0]
                bounds["He"][1] = ran[1]

        return ""

    def shutdown():
        func = request.environ.get("werkzeug.server.shutdown")
        if func is None:
            raise RuntimeError("Not running with the Werkzeug Server")
        func()

    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def display_page(pathname):
        if pathname == "/shutdown":
            shutdown()
        return html.Div([html.H3("{}".format(pathname))])

    app.run_server(debug=False, use_reloader=False)
    return bounds


if __name__ == "__main__":
    cli()
