# Pinhole Analyzer

## Description
Pinhole Analyzer is a program that uses OpenCV to analyze images and count the number of defects on a matrix of pictures. It visualizes the area and number of pinholes per sample and compares the results between samples with boxplots.

## Installation
To use Pinhole Analyzer, you need to clone the project and run it from the source code. Ensure that all necessary libraries are installed by using the `requirements.txt` file.

### Steps:

1. **Source Code**:
   - Clone the repository
   - Install the necessary libraries:
     ```
     pip install -r requirements.txt
     ```
   - Update the folder variable in all of the scripts to point to the main folder containing all the data.
   - Run the scripts in the following order:
     1. `microscope_analyzer_lext_image_separator.py`
     2. `microscope_analyzer_lext.py`
     3. `microscope_analyzer_lext_results_separator.py`

## Usage
Clone the project and run the scripts in the specified order. The program will process images to separate them, analyze pinholes, and organize the results.

Example:
1. Clone the repository to your local machine.
2. Install the necessary libraries using `pip install -r requirements.txt`.
3. Update the folder variable in all scripts to your main data directory.
4. Run the scripts in the following order:
   - `python microscope_analyzer_lext_image_separator.py`
   - `python microscope_analyzer_lext.py`
   - `python microscope_analyzer_lext_results_separator.py`

## Support
For support, you can reach out via the issue tracker on the GitLab repository or contact via email: [support@example.com].

## Roadmap
Future plans for Pinhole Analyzer include:
- Enhancing the defect detection algorithm for more accurate analysis
- Adding support for additional image formats
- Improving the user interface for easier configuration and operation

## Contributing
We welcome contributions! If you'd like to contribute, please fork the repository and create a pull request. Ensure that you follow the contribution guidelines provided in the repository.

To get started:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Create a new Pull Request

## Authors and acknowledgment
Developed by Edgar Nandayapa. Special thanks to all contributors and supporters.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Project status
Active - The project is currently being actively developed and maintained.
