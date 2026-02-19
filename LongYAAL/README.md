# SOFTSEGMENTER

A tool for segmentation of long-form speech translation outputs based on reference segmentation and quality (BLEU) and latency (LongYAAL) evaluation.


## Installation

```bash
# Clone the repository
git clone https://github.com/pe-trik/softsegmenter.git
cd softsegmenter

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python softsegmenter.py \
--hypothesis_file simuleval_instance_file.log \
--ref_sentences_file reference_sentences.txt \
--yaml_file ref_segments.yaml \
--bleu_tokenizer 13a \
--lang en  \
--output_folder segmentation_output
```

Refer to the [example](example/) directory for sample input files and expected output.

## Citation

If you use this tool in your research, please cite it as follows:

```bibtex
@article{polak2025better,
  title={Better Late Than Never: Evaluation of Latency Metrics for Simultaneous Speech-to-Text Translation},
  author={Pol{\'a}k, Peter and Papi, Sara and Bentivogli, Luisa and Bojar, Ond{\v{r}}ej},
  journal={arXiv preprint arXiv:2509.17349},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.