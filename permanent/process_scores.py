def process_scores(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    data = []
    current_header = ''
    current_section = {}

    for line in lines:
        line = line.strip()
        if not line:  # Empty line: end of a section
            if current_section:
                data.append(current_section)
            current_section = {}
        elif '---' in line:  # Header line (model)
            current_header = line.split(' --- ')[0]
            current_section = {'header': current_header, 'subheaders': [], 'BLEU': [], 'chrF2': []}
        elif ':' in line and line.endswith(':'):  # Subheader (language pair)
            subheader = line.strip(':')
            current_section['subheaders'].append(subheader)
        elif 'BLEU =' in line:  # BLEU score
            bleu_score = line.split('=')[1].split()[0].strip()
            current_section['BLEU'].append(bleu_score)
        elif 'chrF2 =' in line:  # chrF2 score
            chrf2_score = line.split('=')[1].strip()
            current_section['chrF2'].append(chrf2_score)

    if current_section:
        data.append(current_section)

    with open(output_file, 'w') as outfile:
        for section in data:
            outfile.write(f"{section['header']}\n")
            outfile.write("subheader\t" + "\t".join(section['subheaders']) + "\n")
            outfile.write("BLEU\t" + "\t".join(section['BLEU']) + "\n")
            outfile.write("chrF2\t" + "\t".join(section['chrF2']) + "\n\n")

if __name__ == "__main__":

    process_scores("../results/nllb-seed/results.txt", "../results/nllb-seed/results_table.txt")
