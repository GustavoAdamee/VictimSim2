import sys
import os
import time

# Adiciona o caminho absoluto para o módulo 'vs' ao sys.path
sys.path.append(os.path.abspath("/home/rudi/Documentos/UTFPR/SI/VictimSim2"))

# Importa classes
from vs.environment import Env
from explorer import Explorer
from rescuer import Rescuer


def main(data_folder_name, config_ag_folder_name):
    # Obtém o diretório atual
    current_folder = os.path.abspath(os.getcwd())
    
    # Resolve os caminhos para as pastas de configuração e dados
    config_ag_folder = os.path.abspath(os.path.join(current_folder, config_ag_folder_name))
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))
    
    # Verifica se os arquivos necessários existem
    size_file = os.path.join(data_folder, "env_config.txt")
    if not os.path.isfile(size_file):
        raise FileNotFoundError(f"Arquivo de configuração do ambiente não encontrado: {size_file}")

    rescuer_file = os.path.join(config_ag_folder, "rescuer_1_config.txt")
    if not os.path.isfile(rescuer_file):
        raise FileNotFoundError(f"Arquivo de configuração do resgatador não encontrado: {rescuer_file}")
    
    # Inicializa o ambiente
    env = Env(data_folder)
    
    # Inicializa o agente resgatador mestre
    master_rescuer = Rescuer(env, rescuer_file, 4)  # 4 é o número de agentes exploradores
    
    # Inicializa os agentes exploradores
    for exp in range(1, 5):
        filename = f"explorer_{exp}_config.txt"
        explorer_file = os.path.join(config_ag_folder, filename)
        if not os.path.isfile(explorer_file):
            raise FileNotFoundError(f"Arquivo de configuração do explorador não encontrado: {explorer_file}")
        Explorer(env, explorer_file, master_rescuer)

    # Executa o simulador de ambiente
    env.run()


if __name__ == '__main__':
    """Para especificar pastas de dados e configuração, passe-as como argumentos de linha de comando.
    Caso contrário, usa os padrões."""
    if len(sys.argv) > 2:
        data_folder_name = sys.argv[1]
        config_ag_folder_name = sys.argv[2]
    else:
        # Caminhos padrão
        data_folder_name = os.path.join("datasets", "data_300v_90x90")
        config_ag_folder_name = os.path.join("ex03_mas_random_dfs", "cfg_1")
    
    # Executa o programa principal
    main(data_folder_name, config_ag_folder_name)
