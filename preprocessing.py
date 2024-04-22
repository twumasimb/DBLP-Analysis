import pickle


def parse_and_save_paper_data(input_file, output_file):
    """
    Parses paper data from a text file and saves it as a pickle file.

    Parameters:
    input_file (str): The path to the text file containing the paper data.
    output_file (str): The path to the output file to save the parsed paper data to.
    """
    def parse_paper_data(data):
        paper = {}
        lines = data.split('\n')

        for line in lines:
            if line.startswith("#*"):
                paper['title'] = line[2:].strip()
            elif line.startswith("#@"):
                paper['authors'] = line[2:].strip().split(', ')
            elif line.startswith("#c"):
                paper['venue'] = line[2:].strip()
            elif line.startswith("#t"):
                paper['year'] = int(line[2:].strip())

        return paper

    with open(input_file, 'r', encoding='utf-8') as file:
        # assuming each paper is separated by two newlines
        papers_data = file.read().split('\n\n')

    papers = [parse_paper_data(paper_data) for paper_data in papers_data]

    # Save as pickle
    with open(output_file, 'wb') as pickle_file:
        pickle.dump(papers, pickle_file)

def save_sample_data(data, file_name: str, start_year: int, end_year: int):
    """
    Filter the input data based on the 'year' key and save the filtered data to a pickle file.

    Args:
        data (list): A list of dictionaries representing the input data.

    Returns:
        None
    """
    sample_data = [item for item in data if 'year' in item and start_year <= item['year'] <= end_year]
    with open(f'{file_name}.pkl', 'wb') as file:
        pickle.dump(sample_data, file)