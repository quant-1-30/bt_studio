export AIRFLOW_HOME=$(pwd)/airflow_home
export PYTHONPATH=$(pwd)
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/bt_studio/pipeline/dags

export AIRFLOW__CORE__LOAD_EXAMPLES=False

rm -f $AIRFLOW_HOME/airflow.db
poetry run airflow db migrate
