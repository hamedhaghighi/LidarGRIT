import argparse
import os

from flask import Flask, render_template, send_file

app = Flask(__name__)

# @app.route('/yaml/<path:filename>')
# def download_file(filename):
#     return send_file(os.path.abspath(filename), as_attachment=True)
def get_last_modified_time(file_path):
    return os.path.getmtime(file_path)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints_dir', type=str, default='', help='Path of the checkpoints directory')
    return parser.parse_args()

@app.route('/yaml/<path:filename>')
def show_yaml_file(filename):
    with open('/' + filename, 'r') as file:
        yaml_content = file.read()
    return render_template('show_yaml.html', yaml_content=yaml_content)

@app.route('/')
def index():
    args = parse_arguments()
    checkpoints_dir = args.checkpoints_dir
    experiments = []

    for root, dirs, files in os.walk(checkpoints_dir):
        for dir_name in dirs:
            experiment_id = dir_name
            yaml_files = [f for f in os.listdir(os.path.join(root, dir_name)) if f.endswith('.yaml')]
            for yaml_file in yaml_files:
                yaml_path = os.path.join(root, dir_name, yaml_file)
                experiments.append({'id': experiment_id, 'yaml_link': yaml_path, 'last_modified': get_last_modified_time(yaml_path)})
    experiments.sort(key=lambda x: x['last_modified'], reverse=True)
    return render_template('index.html', experiments=experiments)

if __name__ == '__main__':
    app.run(debug=True, port=6006)