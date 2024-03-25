## Growth Tracker --> Phenotypic Variables Tracking in Insect Life Cycles

This Python script provides a flexible and powerful tool for tracking and analyzing phenotypic variables throughout the life cycle of insects. It is designed to load data from a CSV file containing detailed information about insect populations, including life cycle data, groups, and phenotypic variables.

### Database Requirements

The CSV file used as input for the script must follow a specific format to ensure that the script can load and analyze the data correctly. The following describes the expected requirements and organization of the database:

1. **CSV File Structure**: The CSV file should have a table structure where each row represents a unique observation of an insect at a specific moment in its life cycle. The columns should include at least the following:

    - **Population**: Unique identifier of the insect population to which the individual belongs.
    - **Cycle**: As reference of generation number.
    - **Group**: Group to which the individual belongs within the population (e.g., control group, experimental group).
    - **Phenotypic Variables**: Additional columns for the phenotypic variables being recorded (e.g., weight, length, area, etc.).
    - **Age**: Age of the individual insect in days.

2. **Comprehensive and Detailed Data**: The CSV file is expected to contain detailed and complete data on the phenotypic observations of the insects throughout their life cycle. The more data provided, the more robust the analysis and visualization that can be performed with the script.

### Using the Script

To use the script, follow these steps:

1. Make sure you have the following Python libraries installed: Pandas, Flask, and Matplotlib. You can install them using pip:

    ```
    pip install pandas flask matplotlib
    ```

2. Prepare your data in a CSV file following the required format described above.

3. Run the script by providing the path to the CSV file as an argument. For example:

    ```
    python script.py data.csv
    ```

4. Once the Flask server is running, open your web browser and navigate to `http://127.0.0.1:5000` to access the web interface.

5. In the web interface, select the desired options for population, cycle, group, and phenotypic variables. Click "Generate Plot" to visualize the graphs and corresponding statistical summary.

### Additional Notes

- Ensure that you organize and structure your data in the CSV file according to the specified requirements to ensure effective analysis and visualization.
- The web interface runs on port 5000 of localhost. Make sure you do not have any other service running on that port.
- Basic knowledge of Python and data manipulation is recommended to effectively understand and utilize this script.

Enjoy detailed visualization and analysis of the phenotypic variables of your insects with this practical and easy-to-use script!
