import os
import re
import requests
import json
import logging
# Configure logging
log_file_name = 'combined_metrics_log.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_name),
        logging.StreamHandler()
    ]
)

OPENAI_API_KEY = os.getenv("RESEARCH_OPENAI_API_KEY", "")

# --- CONFIGURATION DICTIONARY ---
CONFIGURATION = {
    "run_element_accuracy_agent": True,  # Calculate Element Accuracy for Agent Trajectories
    "run_step_success_rate": True,  # Calculate Step Success Rate (Human vs Agent)
    "run_repetitive_action_rate": True,  # Calculate Repetitive Action Rate for Agent Trajectories
    "run_recovery_rate": True,  # Set to True to enable this metric
    "run_partial_success_rate": True,   # Calculate Partial Success Rate for specific tasks

    # Parameters specific to Trajectory Re-alignment Score and other semantic checks
    "semantic_similarity_threshold": 0.7, # Score (0.0-1.0) needed for LLM to consider actions a match
    "partial_success_keywords_file": 'answer_keywords.xlsx',  # File containing keywords for partial success tasks

    "openai_model": "gpt-4o-mini",
    "step_success_direct_match_threshold": 0.94,
    "step_success_sequence_similarity_threshold": 0.7,  # For sequence matches
    "max_agent_sequence_length": 6  # Max steps for agent sequence in Step Success Rate and Initial Alignment Success Rate
}

# --- Helper Functions (Shared) ---
import pandas as pd
import logging


def merge_partial_success_into_output_df(df, partial_success_results, task_col="Task No"):
    """
    Merge partial success results into an existing DataFrame per task.
    Adds columns:
        - 'final_answer'
        - 'total_reference_answers'
        - 'matched_reference_answers'
        - 'score'

    If a task does not exist in df, a new row is added.

    Parameters:
        df: pd.DataFrame → original DataFrame with Task No column
        partial_success_results: dict → output from partial_success()
        task_col: str → name of the task ID column in df
    Returns:
        pd.DataFrame with added columns
    """

    # Define new columns
    new_cols = ["final_answer", "total_reference_answers", "matched_reference_answers", "score"]
    for col in new_cols:
        if col not in df.columns:
            df[col] = None

    for task_no, result in partial_success_results.items():
        # If task exists, update values
        if task_no in df[task_col].values:
            for col in new_cols:
                df.loc[df[task_col] == task_no, col] = result.get(col)
        else:
            # Add new row if task not in df
            new_row = {task_col: task_no}
            for col in new_cols:
                new_row[col] = result.get(col)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    logging.info(f"Partial success results merged into DataFrame with {len(df)} tasks.")
    return df


def merge_recovery_results_into_output_df(df, recovery_results, task_col="Task No"):
    """
    Merge recovery results (dict) into an existing DataFrame based on Task No.
    Adds new rows if tasks are not already present.

    Parameters:
        df (pd.DataFrame): The existing DataFrame with a Task No column.
        recovery_results (dict): {task_no: metrics_dict}
        task_col (str): Column name for task IDs.
    """
    # Ensure Task No column exists
    if task_col not in df.columns:
        df[task_col] = None

    # Convert results dict -> DataFrame
    recovery_df = pd.DataFrame.from_dict(recovery_results, orient="index").reset_index()
    recovery_df.rename(columns={"index": task_col}, inplace=True)

    # Merge on Task No
    df = pd.merge(df, recovery_df, on=task_col, how="outer")

    return df
'''def merge_repetetive_results_into_output_df(df, results, task_col="Task No"):
    """
    Merge the repetitive action results into the DataFrame.

    Args:
        df: pandas DataFrame containing at least a column with task ids
        results: dict from repetitive_action_score()
        task_col: column name in df that holds task ids
    Returns:
        df with new columns: same_actions_count, same_actions, total_actions, score
    """
    # Make sure the new columns exist
    new_cols = ["same_actions_count", "same_actions", "total_actions", "repetetive_score"]
    for col in new_cols:
        if col not in df.columns:
            df[col] = None

    # Update existing rows or add new rows
    for task_no, data in results.items():
        # Check if task exists
        if task_no in df[task_col].values:
            # Update row
            for col in new_cols:
                value = data.get("llm_response", {}).get(col, data.get(col))
                if isinstance(value, list):
                    value = str(value)  # convert list to string to store in one cell
                df.loc[df[task_col] == task_no, col] = value

        else:
            # Add new row
            new_row = {task_col: task_no}
            for col in new_cols:
                new_row[col] = data.get("llm_response", {}).get(col, data.get(col))
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return df
    '''
def merge_repetetive_results_into_output_df(df, results, task_col="Task No"):
    """
    Merge the repetitive action results into the DataFrame.

    Args:
        df: pandas DataFrame containing at least a column with task ids
        results: dict from repetitiveness()
        task_col: column name in df that holds task ids
    Returns:
        df with new columns: same_actions_count, same_actions, total_actions, score
    """
    # Match the keys in your results dict
    new_cols = ["same_actions_count", "same_actions", "total_actions", "repetetive_score"]

    # Make sure the new columns exist in df
    for col in new_cols:
        if col not in df.columns:
            df[col] = None

    # Update existing rows or add new rows
    for task_no, data in results.items():
        # Ensure values are aligned with results dict
        row_values = {col: data.get(col) for col in new_cols}

        # Convert lists to strings for Excel/CSV friendliness
        if isinstance(row_values["same_actions"], list):
            row_values["same_actions"] = str(row_values["same_actions"])

        if task_no in df[task_col].values:
            # Update row
            for col, value in row_values.items():
                df.loc[df[task_col] == task_no, col] = value
        else:
            # Add new row
            new_row = {task_col: task_no, **row_values}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return df
def extract_action_from_text(text_string):
    """
    Extracts the action from a string.
    e.g: extracts "type [35] [Randyland] [1]" from "In summary, the next action I will perform is ```type [35] [Randyland] [1]```"
    If multiple such phrases exist, only the last occurrence is considered.
    """
    if not isinstance(text_string, str):
        return None
    # Find all non-overlapping matches and take the last one
    matches = re.findall(r'```(.*?)```', text_string, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Return the last match
    return None

def replace_agentDF_with_readable_actions(df, api_key, model_name):
    """
    Creates a copy of the input DataFrame and transforms each cell in the 'Text' column
    using the `transform_agent_output_to_readable_step` function.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'Text' column.
        api_key (str): OpenAI API key.
        model_name (str): LLM model name to use.

    Returns:
        pd.DataFrame: A new DataFrame copy with transformed 'Text' column.
    """
    required_columns = ['Task No', 'Step No', 'Text', 'Next Step']
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"Input DataFrame must contain a '{col}' column.")
            return None

    # Copy only the required columns
    df_copy = df[required_columns].copy()

    # Apply transformation to each row in the 'Text' column
    df_copy['Text'] = df_copy['Text'].apply(
        lambda x: transform_agent_output_to_readable_step(str(x), api_key, model_name) if pd.notna(x) else x
    )

    return df_copy
# ---  Functions with LLM Interaction  ---


def transform_agent_output_to_readable_step(agent_action, api_key, model_name):
    """
    Extracts the essence of the next agent action: the action type and what it is applied on.

    Args:
        agent_action (str): The agent's plan or description of the next action.
        api_key (str): OpenAI API key.
        model_name (str): LLM model to use.

    Returns:
        str: A concise description of the agent's next action (action + target).
    """
    if not api_key:
        logging.error("OpenAI API key not set for extracting agent action essence.")
        return None

    # Prompt with clear instructions and examples
    prompt = f"""
You are an assistant that reads an agent's planned action description and extracts the **essence**: 
what action will be performed, and what object or element it is applied on. 
Return a concise and human-readable action. Do not include anything else.

Examples:

Input: Let's think step-by-step. The previous action was to enter the origin: Carnegie Mellon University. Now, I need to type the destination. I will type "Randyland" in the 'To' search box to indicate the destination . 
In summary, the next action I will perform is ```type [559] [Randyland] [1]```
Output: Type "Randyland" in the 'To' search box

Input: Let's think step-by-step. To find out how many fulfilled orders I have over the past month and the total amount spent, I need to navigate to the "My Account" section. From the current page, I see a link for "My Account" which I can click on to access order details.
In summary, the next action I will perform is ```click [1177]```
Output: Click "My Account" link

Now, extract the essence of this next action:

Input: {agent_action}
Output:
"""

    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    messages = [
        {"role": "system", "content": "You are a helpful assistant that extracts the essence of agent actions."},
        {"role": "user", "content": prompt}
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.0
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        # Extract assistant's content
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return content
    except Exception as e:
        logging.error(f"Error calling OpenAI API for action extraction: {e}")
        return None


def compare_agent_step_to_human(human_intended_action, agent_text_sequence, api_key, model_name):
    """
    Compares an extracted human intended action to an agent step.
    Returns a score between 0 and 1.
    """
    if not api_key:
        logging.error("OpenAI API key not set for step success rate fulfillment score.")
        return 0.0

    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    messages = [
        {"role": "system",
         "content": "You are an expert Action Sequence Evaluator."},
        {"role": "user", "content": f"""
        You will be provided with two actions: 1. Human Recorded Action (HRA): [The human's action description] 2. Agent-Recorded Action (ARA): [The agent's action description, which may include metadata like IDs]. Your task is to determine  whether the Human-Recorded Action and the Agent-Recorded Action are semantically equivalent.
        Semantic Equivalence means both actions convey the same intent.
        Your output must be a single numerical value:
        1 if both actions are semantically identical, 0 if they are not.
        Note:
        1. Minor textual variations such as differences in casing ('Design' vs 'design'), extra/missing spaces, or punctuation should be ignored as long as the core identity and meaning are preserved.
        2. Any substantive mismatch in meaning or details (e.g., 'Byte Blaze' vs 'Lukas Opp', or different dates) must result in a 0
        Evaluation Steps: 1. Read the Human-Recorded Action and the Agent-Recorded Action carefully. 2. Rate the output as 1 (same) or 0 (different) according to the criteria above.
        Your response must be a JSON object with a single key 'fulfillmentScore' and its numerical value. Do not include any other text or explanation outside the JSON.

        Human Intended Action: "{human_intended_action}"
        Agent Action(s): "{agent_text_sequence}"
        """}
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.0
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        content = result.get('choices', [{}])[0].get('message', {}).get('content')
        if content:
            parsed_json = json.loads(content)
            score = parsed_json.get('fulfillmentScore')
            if score is not None and isinstance(score, (int, float)):
                return float(score)
        logging.warning(f"OpenAI response did not contain a valid 'fulfillmentScore'. Raw response: {content}")
        return 0.0
    except (requests.exceptions.RequestException, json.JSONDecodeError, Exception) as e:
        logging.error(f"Error calling OpenAI API for step success rate: {e}")
        return 0.0


def compare_current_to_next_step(intended_action, actual_action, api_key, model_name):
    """
    Compares an intended action string with an "actual action" string using OpenAI
    and returns a binary score (1 for match, 0 for mismatch) for Element Accuracy.
    """
    if not api_key:
        logging.error("OpenAI API key not set for element accuracy.")
        return 0

    if not intended_action or not actual_action:
        return 0

    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    messages = [
        {"role": "system",
         "content": "You are a strict validator checking the internal consistency of an agent's actions. Your task is to determine if the 'Actual Action' precisely matches the 'Intended Action' semantically and functionally, returning a binary score (1 for match, 0 for mismatch). You must pay close attention to action type and primary ID numbers."},
        {"role": "user", "content": f"""
        **Instructions for scoring:**
        * A score of 1 indicates a perfect functional match.
        * A score of 0 indicates a functional mismatch.
        * **Match for Action Type and Primary ID**: The command type (e.g., 'click', 'type') and the primary element identifier (e.g., a numerical ID like `[35]`) must functionally match. Case sensitivity for commands is not required; 'click' and 'CLICK' are considered a match. The ID's value itself must match.
        * **Ignore cosmetic differences**: Disregard variations in trailing spaces within parameters (e.g., `[Randyland]` vs `[Randyland ]`), or casing in element names (e.g., `Design` vs `design`) if the core identity is preserved.
        * **Ignore optional parameters**: If the 'Intended Action' contains an optional parameter (like `[1]` in `type [ID] [value] [optional_param]`) that is absent in the 'Actual Action', but the core command and its primary parameters match, still score 1. Assume such an optional parameter does not change the core functional intent if omitted.
        * **Ignore additional descriptive text**: If the 'Actual Action' contains extra descriptive text (e.g., 'where [ID] is [text] link/button/textbox', 'focused: True', 'required: False') that is not present in the 'Intended Action', but the core command and primary parameters align, still score 1. This extra detail does not alter the functional match.

        Your response must be a JSON object with a single key 'match' and its numerical value (1 or 0). Do not include any other text or explanation outside the JSON.

        Intended Action: "{intended_action}"
        Actual Action: "{actual_action}"
        """}
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.0
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        content = result.get('choices', [{}])[0].get('message', {}).get('content')
        if content:
            parsed_json = json.loads(content)
            score = parsed_json.get('match')
            if score is not None and isinstance(score, (int, float)):
                return int(score)
        logging.warning(f"OpenAI response did not contain a valid 'match' score. Raw response: {content}")
        return 0
    except (requests.exceptions.RequestException, json.JSONDecodeError, Exception) as e:
        logging.error(f"Error calling OpenAI API for element accuracy: {e}")
        return 0

def compare_partial_answers(question, reference_keywords, agent_answer, api_key, model_name):
    """
    Uses the LLM to check if each reference keyword/phrase is fulfilled by the agent's final answer.
    Returns a dictionary:
        {
            "keyword1": True/False,
            "keyword2": True/False,
            ...
        }
    """
    if not api_key:
        logging.error("OpenAI API key not set for partial success keyword matching.")
        return {kw: False for kw in reference_keywords}

    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}


    # Build structured prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are an evaluator. Your job is to check if each reference keyword/phrase "
                "is present or semantically satisfied in the student's answer. "
                "Answer in JSON only, with keys equal to the keywords and values True/False."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task:\n{question}\n\n"
                f"Reference keywords/phrases: {reference_keywords}\n\n"
                f"Agent's final answer: {agent_answer}\n\n"
                "Return JSON only. Example:\n"
                "{ \"keyword1\": true, \"keyword2\": false }"
            ),
        },
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "response_format": {"type": "json_object"},  # enforce JSON output
        "temperature": 0.0,
    }
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        content = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logging.warning(
                f"OpenAI response was not valid JSON. Raw response: {content}. Falling back to all False."
            )
            return {kw: False for kw in reference_keywords}

    except (requests.exceptions.RequestException, Exception) as e:
        logging.error(f"Error calling OpenAI API for partial success: {e}. Falling back to all False.")
        return {kw: False for kw in reference_keywords}






# --- Metric Calculation Functions (Operating on DataFrames) ---
def element_accuracy(df, api_key, model_name, df_name="DataFrame"):
    """
    Calculates the Element Accuracy for each step in a given DataFrame.
    Adds 'Element Accuracy Score' column.
    If an "Early stop" is found at step N within a task, only steps 1 to N-1 are considered for that task.
    """
    logging.info(f"\n--- Calculating Element Accuracy for {df_name} ---")

    # Ensure required columns exist
    required_columns = ['Task No', 'Step No', 'Text', 'Next Step']
    if not all(col in df.columns for col in required_columns):
        logging.error(f"{df_name} must contain columns: {required_columns}.")
        return None

    df_copy = df.copy()  # Work on a copy to avoid modifying the original DataFrame
    df_result = pd.DataFrame(columns=df_copy.columns.tolist() + ['Element Accuracy Score'])  # To store results

    processed_tasks_info = {}  # To keep track of which tasks were truncated

    for task_no, task_df in df_copy.groupby('Task No'):
        # Sort steps by 'Step No'
        task_df = task_df.sort_values('Step No').reset_index(drop=True)

        # Default: no early stop
        valid_steps_for_task = task_df.copy()
        processed_tasks_info[task_no] = "No Early stop found. Calculating EA for all steps."

        # Check for early stop
        for idx, row in task_df.iterrows():
            if isinstance(row['Next Step'], str) and 'early stop' in row['Next Step'].lower():
                first_early_stop_step_no = row['Step No']
                valid_steps_for_task = task_df[task_df['Step No'] < first_early_stop_step_no].copy()
                processed_tasks_info[task_no] = (
                    f"Early stop at Step {first_early_stop_step_no}. "
                    f"Calculating EA for steps 1 to {first_early_stop_step_no - 1}."
                )
                break  # stop at first early stop


        # Initialize 'Element Accuracy Score' for the valid steps of this task
        if not valid_steps_for_task.empty:
            # Do scoring
            valid_steps_for_task['Element Accuracy Score'] = 0.0

            # Perform Element Accuracy calculation on valid steps
            for index, row in valid_steps_for_task.iterrows():
                task_no_current = row['Task No']  # Ensure task_no is consistent if needed
                step_no = row['Step No']
                text_instruction = row['Text']
                next_step_action = row['Next Step']

                intended_action_parsed = extract_action_from_text(text_instruction)

                if not isinstance(next_step_action, str):
                    next_step_action = str(next_step_action) if pd.notna(next_step_action) else None

                if intended_action_parsed and next_step_action:
                    score = compare_current_to_next_step(intended_action_parsed, next_step_action.strip(), api_key,
                                                         model_name)
                    valid_steps_for_task.at[index, 'Element Accuracy Score'] = score
                    logging.info(
                        f"  {df_name} - Task {task_no_current}, Step {step_no}: Intended: '{intended_action_parsed}', Actual: '{next_step_action}'. Score: {score}")
                else:
                    logging.warning(
                        f"  {df_name} - Task {task_no_current}, Step {step_no}: Could not extract intended action or 'Next Step' is missing/invalid. Text: '{text_instruction}', Next step: '{next_step_action}'. Score: 0")
                    valid_steps_for_task.at[index, 'Element Accuracy Score'] = 0

            # Append the results for this task to the overall result DataFrame
            df_result = pd.concat([df_result, valid_steps_for_task])
        else:
            logging.info(f"  Task {task_no}: No valid steps to calculate Element Accuracy after early stop truncation.")

    logging.info("\n--- Element Accuracy Calculation Summary per Task ---")
    for task_no, info in processed_tasks_info.items():
        logging.info(f"Task {task_no}: {info}")
    logging.info("---------------------------------------------------\n")

    correct_steps = (df_result['Element Accuracy Score'] == 1.0).sum()
    if valid_steps_for_task.shape[0] > 0:
        element_accuracy_rate= correct_steps / valid_steps_for_task.shape[0]
    else:
        element_accuracy_rate = None

    return element_accuracy_rate



def step_success(human_df, agent_df, api_key, model_name):
    """
    Compares full human and agent trajectories using OpenAI.
    Calculates the percentage of agent steps that are part of the human trajectory
    and are not redundant.
    Score = (#matching agent steps) / (#human steps).
    Returns a dictionary with task_no -> result dict.
    """
    logging.info("\n--- Calculating Step Success Rates ---")

    if not api_key:
        logging.error("OpenAI API key not set for extracting agent action essence.")
        return {}

    # Group by Task No
    human_tasks = human_df.groupby('Task No')
    agent_tasks = agent_df.groupby('Task No')

    results = {}

    common_tasks = set(human_tasks.groups.keys()).intersection(set(agent_tasks.groups.keys()))
    if not common_tasks:
        logging.warning("No common tasks found between human and agent files for step success rate calculation.")
        return {}

    for task_no in sorted(common_tasks):
        logging.info(f"\n--- Processing Step Success Rate for Task No: {task_no} ---")

        # Collect steps
        human_steps = human_tasks.get_group(task_no).sort_values('Step No')['Text'].tolist()
        agent_steps = agent_tasks.get_group(task_no).sort_values('Step No')['Text'].tolist()

        # Build the prompt
        prompt = f"""
        You are comparing two trajectories: agent steps and human steps, in **meaning**, even if the wording is slightly different.
        For example: "Type hello in the text search box" and "Type hello in the search box" are the same action.
        Rules to follow:
        1. Each human step can match exactly once with an agent step.
        2. If the agent repeats a step more times than the human, ignore it because you have already counted the first occurance. 
        3. If a human repeats an action, the agent must repeat it the same number of times.
        
        Example 1: 
        Human Trajectory: Add item A to cart, Add item A to cart. 
        Agent Trajectory: Add item A to cart. 
        Output: matching_steps: ["Add item A to cart"] (Agent is missing one repetition)
        
        Example 2:
        Human Trajectory: Add item B to cart. 
        Agent Trajectory: Add item B to cart, Add item B to cart. 
        Output: matching_steps: ["Add item B to cart"] (Human performed the step one time)

        Human steps:
        {human_steps}

        Agent steps:
        {agent_steps}

        Return the result in this JSON format:
        {{
          "matching_steps": [list of matching agent steps according to the rules above]
        }}
        """

        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        messages = [
            {"role": "system", "content": "You are a strict evaluator of task trajectories."},
            {"role": "user", "content": prompt}
        ]
        payload = {"model": model_name, "messages": messages, "temperature": 0.0}

        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            clean_content = re.sub(r"^```json\s*|\s*```$", "", content, flags=re.DOTALL)
            # Parse JSON safely
            result = json.loads(clean_content)

            num_matching = len(result.get("matching_steps", []))
            result["score_sr"] = num_matching / len(human_steps) if len(human_steps) > 0 else 999

            result["total_human_steps_sr"] = len(human_steps)
            result["total_agent_steps_sr"] = len(agent_steps)
            result["matches"] = num_matching
            results[task_no] = result

            logging.info(
                f"Task {task_no}: Matches {num_matching} / Total Human Steps: {len(human_steps)} / Total agent Steps {len(agent_steps)}"
                f"-> Score {result['score_sr']:.2f} | Matching steps: {result.get('matching_steps', [])}"
            )

        except Exception as e:
            logging.error(f"Failed to process Task {task_no}: {e}")
            results[task_no] = None

    return results


def repetitiveness(agent_df, api_key, model_name):
    """
    Calculates repetitive actions in a task according to 'strictly previous' rule.
    Sends the actions to LLM to identify repeated actions.
    Score = 1 - (number of repetitive actions / total actions), calculated locally.
    Returns: task_no -> dict with LLM answer, total actions, score.
    """
    import logging
    import requests
    import json
    import re

    logging.info("\n--- Calculating Repetitive Action Scores ---")

    if not api_key:
        logging.error("OpenAI API key not set for extracting repetitive actions.")
        return {}

    agent_tasks = agent_df.groupby('Task No')
    results = {}

    for task_no in sorted(agent_tasks.groups.keys()):
        logging.info(f"\n--- Processing Task No: {task_no} ---")

        actions = agent_tasks.get_group(task_no).sort_values('Step No')['Text'].tolist()
        total_actions = len(actions)

        if total_actions == 1:
            results[task_no] = {
                #"llm_response": None,
                "total_actions": total_actions,
                "repetetive_score": 1.0,
                "same_actions_count": 0,
                "same_actions": []
            }
            logging.info(
                f"  Task {task_no}: Total actions = {total_actions}. No repetitive actions possible. Score: 1.00")
            continue

        prompt = f"""
        You are given a sequence of agent actions for a single task:
        {actions}
        Identify all **adjacent repetitive actions**.

        Rules:
        1. A repetitive action is one that repeats the previous action in **meaning**, even if the wording is slightly different.
        For example: "Type hello in the text search box" and "Type hello in the search box" are the same action.
        2. Only count repetitions that are **adjacent** in the sequence.  
        3. Ignore actions that are similar but not immediately following the previous action. 
        Only consider repetitions of the same action. For example, for the sequence [A, B, B, C, A]:
          - The second B is repetitive
          - The second A is *not* repetitive

        Return the result in JSON format:
        {{
            "same_actions_count": <number of repetitive actions>,
            "same_actions": [list of repeated actions as strings]
        }}
        """

        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        messages = [
            {"role": "system", "content": "You are an expert at analyzing sequences of actions."},
            {"role": "user", "content": prompt}
        ]
        payload = {"model": model_name, "messages": messages, "temperature": 0.0}

        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            clean_content = re.sub(r"^```json\s*|\s*```$", "", content, flags=re.DOTALL)
            llm_result = json.loads(clean_content)

            same_actions_count = llm_result.get("same_actions_count", 0)
            repetetive_score = 1 - (same_actions_count / total_actions)

            results[task_no] = {
                #"llm_response": llm_result,
                "total_actions": total_actions,
                "same_actions": llm_result.get("same_actions", []),
                "same_actions_count": same_actions_count,
                "repetetive_score": round(repetetive_score, 3)
            }

            logging.info(
                f"  Task {task_no}: Total actions = {total_actions}, "
                f"Repetitive actions = {same_actions_count}, Score = {repetetive_score:.3f}, "
                f"Repeated actions: {llm_result.get('same_actions', [])}"
            )

        except Exception as e:
            logging.error(f"Failed to process Task {task_no}: {e}")
            results[task_no] = {
                "total_actions": total_actions,
                "same_actions": [],
                "same_actions_count": 0,
                "repetetive_score": None
            }

    return results



def recovery(human_df, agent_df, api_key, model_name, semantic_similarity_threshold,
             max_agent_sequence_length):
    """
    Measures the agent's ability to recover from deviations from the human trajectory.
    Calculates Recovery Success Rate and Average Recovery Cost per task.

    Args:
        human_df (pd.DataFrame): DataFrame with human task data ('Task No', 'Step No', 'Text').
        agent_df (pd.DataFrame): DataFrame with agent task data ('Task No', 'Step No', 'Text').
        api_key (str): OpenAI API key for LLM calls.
        model_name (str): Name of the LLM model to use.
        semantic_similarity_threshold (float): Score required for a semantic match (0.0-1.0).
        max_agent_sequence_length (int): Max number of agent steps to group for comparison.

    Returns:
        dict: A dictionary where keys are 'Task No' and values are dictionaries
              containing 'Recovery_Success_Rate', 'Average_Recovery_Cost_Steps',
              'Total_Deviation_Incidents', 'Total_Successful_Recoveries',
              'Total_Recovery_Cost_Steps'.
              Returns an empty dict if no common tasks.
    """
    logging.info(f"\n--- Calculating Trajectory Re-alignment Score ---")

    required_cols = ['Task No', 'Text']
    if not all(col in human_df.columns for col in required_cols) or \
            not all(col in agent_df.columns for col in required_cols):
        logging.error(f"Human and Agent DataFrames must contain columns: {required_cols}.")
        return {}

    realignment_results = {}

    human_tasks = human_df.groupby('Task No')
    agent_tasks = agent_df.groupby('Task No')

    common_tasks = set(human_tasks.groups.keys()).intersection(set(agent_tasks.groups.keys()))

    if not common_tasks:
        logging.warning(
            "No common tasks found between human and agent files for trajectory re-alignment score calculation. Returning empty results.")
        return {}  # Return empty dictionary as no task-specific scores can be computed

    for task_no in sorted(list(common_tasks)):
        logging.info(f"  Task {task_no}: Analyzing trajectory re-alignment.")

        human_steps_raw = human_tasks.get_group(task_no).sort_values('Step No')['Text'].tolist()
        human_actions = [text.strip() for text in human_steps_raw if pd.notna(text) and text.strip() != '']

        agent_steps_raw = agent_tasks.get_group(task_no).sort_values('Step No')['Next Step'].tolist()
        agent_actions = [text.strip() for text in agent_steps_raw if pd.notna(text) and text.strip() != '']

        # Initialize state and counters for the current task
        human_ptr = 0
        agent_ptr = 0
        deviation_status = "on-track"  # "on-track" or "deviating"
        num_deviation_incidents = 0
        num_successful_recoveries = 0
        total_recovery_cost_steps = 0  # Agent steps taken during deviation before recovery
        deviation_start_agent_step_index = -1  # Index of agent action where current deviation began

        # ... (edge case checks for empty human_actions or agent_actions lists - these remain the same)

        while human_ptr < len(human_actions) and agent_ptr < len(agent_actions):
            current_human_target = human_actions[human_ptr]

            # --- PHASE 1: Check if the current agent action directly matches the current human target ---
            current_agent_action_for_deviation_check = agent_actions[agent_ptr]
            score_current_agent_vs_human_target = compare_agent_step_to_human(
                current_human_target,
                current_agent_action_for_deviation_check,
                api_key,
                model_name
            )

            is_directly_matching_current_human = (score_current_agent_vs_human_target >= semantic_similarity_threshold)

            # --- PHASE 2: Handle deviation status transition and pointer advancement ---
            if not is_directly_matching_current_human:
                # If agent's current step does not directly match the human target:
                if deviation_status == "on-track":
                    # This is the first action that causes a deviation.
                    num_deviation_incidents += 1
                    deviation_status = "deviating"
                    deviation_start_agent_step_index = agent_ptr
                    logging.warning(
                        f"    Task {task_no}: Deviation initiated! Agent step {agent_ptr + 1} ('{current_agent_action_for_deviation_check}') did not directly fulfill Human step {human_ptr + 1} ('{current_human_target}').")

                # Agent's action did not fulfill the current human step directly.
                # Consume this agent step as part of the deviation (or continued deviation).
                agent_ptr += 1
                logging.info(
                    f"    Task {task_no}: Agent advances to next action ({agent_ptr}). Human step {human_ptr + 1} remains unfulfilled.")
                # Continue the loop to re-evaluate the SAME human_ptr against the NEXT agent_ptr
                continue

                # --- PHASE 3: If current agent step *does* directly match current human target ---
            else:  # is_directly_matching_current_human is True
                if deviation_status == "deviating":
                    # We were deviating, but now we have a direct match -> Recovery!
                    num_successful_recoveries += 1
                    # Recovery cost: steps from deviation start to current agent_ptr (inclusive)
                    deviation_length = (agent_ptr + 1) - deviation_start_agent_step_index
                    total_recovery_cost_steps += deviation_length
                    logging.info(
                        f"      Recovery detected! Direct match for Human step {human_ptr + 1} by Agent step {agent_ptr + 1}. Cost: {deviation_length} agent steps.")
                    deviation_status = "on-track"
                    deviation_start_agent_step_index = -1  # Reset

                # Advance both pointers as the human step is fulfilled by a direct match.
                human_ptr += 1
                agent_ptr += 1
                logging.info(
                    f"    Task {task_no}: Human step {human_ptr} fulfilled by direct match. Moving to next human step.")
        # --- After loop: Finalize metrics for the current task ---
        recovery_success_rate = 0.0
        if num_deviation_incidents > 0:
            recovery_success_rate = num_successful_recoveries / num_deviation_incidents
        # If no deviations occurred, and all human steps were fulfilled, 100% success rate
        elif human_ptr == len(human_actions):
            recovery_success_rate = 1.0

        avg_recovery_cost_steps = 0.0
        if num_successful_recoveries > 0:
            avg_recovery_cost_steps = total_recovery_cost_steps / num_successful_recoveries
        elif num_deviation_incidents > 0:
            # If there were deviations but no successful recoveries, average cost is infinite
            avg_recovery_cost_steps = float('inf')

        # Handle remaining unfulfilled human steps when agent trajectory ends
        if human_ptr < len(human_actions):
            if deviation_status == "on-track":
                # Agent ran out of actions while on-track, but human steps remain.
                # Each remaining human step implies a new deviation that couldn't recover.
                num_deviation_incidents += (len(human_actions) - human_ptr)
            # If deviation_status was "deviating" and agent_ptr ran out,
            # that existing deviation incident is already counted and did not recover.

            # Recalculate success rate if new incidents were added at the end
            if num_deviation_incidents > 0:
                recovery_success_rate = num_successful_recoveries / num_deviation_incidents
            # If no successful recoveries but incidents, cost remains inf
            if num_successful_recoveries == 0 and num_deviation_incidents > 0:
                avg_recovery_cost_steps = float('inf')

        logging.info(f"    Task {task_no} Summary:")
        logging.info(f"      Total Deviation Incidents: {num_deviation_incidents}")
        logging.info(f"      Total Successful Recoveries: {num_successful_recoveries}")
        logging.info(f"      Total Recovery Cost (Agent Steps): {total_recovery_cost_steps}")
        logging.info(f"      Recovery Success Rate: {recovery_success_rate:.2f}")
        logging.info(f"      Average Recovery Cost (Agent Steps): {avg_recovery_cost_steps:.2f}")

        realignment_results[task_no] = {
            'Recovery_Success_Rate': recovery_success_rate,
            'Average_Recovery_Cost_Steps': avg_recovery_cost_steps,
            'Total_Deviation_Incidents': num_deviation_incidents,
            'Total_Successful_Recoveries': num_successful_recoveries,
            'Total_Recovery_Cost_Steps': total_recovery_cost_steps
        }

    logging.info(f"--- Finished Trajectory Re-alignment Score calculation ---")
    return realignment_results

def partial_success(agent_df, keywords_df, api_key, model_name):
    """
    Measures how well the agent’s final answer matches the expected reference answers per task.
    Uses LLM semantic similarity only (via compare_partial_answers).

    Output (dict per task):
    {
        task_no: {
            "final_answer": str,
            "total_reference_answers": int,
            "matched_reference_answers": int,
            "score": float,
            "results": {ref1: True/False, ref2: True/False, ...}
        },
        ...
    }
    """

    logging.info("\n--- Calculating Partial Success Rate ---")

    # 1. Input validation
    required_agent_cols = ['Task No', 'Step No', 'Next Step']
    required_keywords_cols = ['Task No', 'Answer Must Include']

    if not all(col in agent_df.columns for col in required_agent_cols):
        raise ValueError(f"Agent DataFrame must contain columns: {required_agent_cols}")

    if not all(col in keywords_df.columns for col in required_keywords_cols):
        raise ValueError(f"Keywords DataFrame must contain columns: {required_keywords_cols}")

    results_per_task = {}

    # 2. Prepare keywords per task
    reference_map = (
        keywords_df.set_index('Task No')['Answer Must Include']
        .apply(lambda x: [kw.strip() for kw in str(x).split(',') if kw.strip()])
        .to_dict()
    )

    # 3. Get agent’s final answer per task
    agent_last_steps = agent_df.groupby('Task No').apply(
        lambda x: x.sort_values('Step No').iloc[-1]
    )

    # 4. Evaluate only tasks present in both agent_df and keywords_df
    tasks_to_evaluate = set(agent_last_steps.index).intersection(reference_map.keys())
    for task_no in tasks_to_evaluate:
        ref_keywords = reference_map[task_no]
        agent_final_answer = str(agent_last_steps.loc[task_no, 'Next Step']).strip()

        # Call LLM to check each reference answer
        per_keyword_results = compare_partial_answers(
            question=f"Task {task_no}",
            reference_keywords=ref_keywords,
            agent_answer=agent_final_answer,
            api_key=api_key,
            model_name=model_name,
        )

        # Count matches
        matched = sum(1 for v in per_keyword_results.values() if v)
        total = len(ref_keywords)
        score = matched / total if total > 0 else 1.0

        # Store results
        results_per_task[task_no] = {
            "final_answer": agent_final_answer,
            "total_reference_answers": total,
            "matched_reference_answers": matched,
            "score": score,
            "results": per_keyword_results,
        }

        logging.info(
            f"  Task {task_no}: Final='{agent_final_answer}' "
            f"Matched {matched}/{total} -> Score={score:.2f}"
        )

    return results_per_task

# --- Main Execution Block ---

if __name__ == "__main__":
    logging.info("Starting evaluation script...")

    human_excel_path = 'tasks_cleaned.xlsx'
    agent_excel_path = 'processed_output\\merged_highlighted_texts_local.xlsx'
    keywords_excel_path = CONFIGURATION["partial_success_keywords_file"]

    human_df = None
    agent_df = None
    keywords_df = None

    try:
        human_df = pd.read_excel(human_excel_path)
        logging.info(f"Loaded human data from {human_excel_path}. Shape: {human_df.shape}")
        # logging.debug(f"Human DataFrame head:\n{human_df.head()}") # Keep commented unless debugging
    except FileNotFoundError:
        logging.error(f"Error: Human data file not found at {human_excel_path}")
    except Exception as e:
        logging.error(f"Error loading human data Excel: {e}")

    try:
        agent_df = pd.read_excel(agent_excel_path)
        logging.info(f"Loaded agent data from {agent_excel_path}. Shape: {agent_df.shape}")
        # logging.debug(f"Agent DataFrame head:\n{agent_df.head()}") # Keep commented unless debugging
    except FileNotFoundError:
        logging.error(f"Error: Agent data file not found at {agent_excel_path}")
    except Exception as e:
        logging.error(f"Error loading agent data Excel: {e}")


    try:
        keywords_df = pd.read_excel(keywords_excel_path)
        logging.info(f"Loaded keywords from {keywords_excel_path}. Shape: {keywords_df.shape}")
        # logging.debug(f"Keywords DataFrame head:\n{keywords_df.head()}") # Keep commented unless debugging
    except FileNotFoundError:
        logging.error(f"Error: Keywords file not found at {keywords_excel_path}")
    except Exception as e:
        logging.error(f"Error loading keywords Excel: {e}")

    readable_agent_df = replace_agentDF_with_readable_actions(agent_df,OPENAI_API_KEY, CONFIGURATION["openai_model"])
    readable_agent_df.to_csv("readable_agent_traj.csv", mode='a', header=False, index=False)
    logging.info(f"Converted agent trajectories to readable format")

    task_numbers = agent_df['Task No'].unique() if agent_df is not None else []
    all_task_metrics = pd.DataFrame({"Task No": task_numbers})

    if CONFIGURATION["run_element_accuracy_agent"] and agent_df is not None:
       """ logging.info("Starting Element Accuracy calculation...")

        unique_agent_tasks = agent_df['Task No'].unique()

        for task_no in sorted(list(unique_agent_tasks)):
            agent_task_df = agent_df[agent_df['Task No'] == task_no].copy().reset_index(drop=True)

            if agent_task_df.empty:
                logging.warning(f"Skipping Task {task_no} for Element Accuracy due to empty agent data.")
                continue

            metrics = element_accuracy(
                df=agent_task_df,
                api_key=OPENAI_API_KEY,
                model_name=CONFIGURATION["openai_model"],
                df_name="DataFrame"
            )"""

       logging.info("Starting Element Accuracy calculation...")
       element_accuracy_scores = {}
       for task_no in sorted(task_numbers):
           task_df = agent_df[agent_df['Task No'] == task_no].copy().reset_index(drop=True)
           if task_df.empty:
               logging.warning(f"Task {task_no} skipped: empty agent data.")
               continue
           score = element_accuracy(task_df, OPENAI_API_KEY, CONFIGURATION["openai_model"])
           element_accuracy_scores[task_no] = score
           logging.info(f"Task {task_no}: Element Accuracy = {score:.2f}")
       all_task_metrics['Element Accuracy'] = all_task_metrics['Task No'].map(element_accuracy_scores)

    else:
        logging.info("Skipping Element Accuracy calculation.")
        all_task_metrics['Element Accuracy'] = pd.Series(dtype=float)


    if CONFIGURATION["run_step_success_rate"] and human_df is not None and agent_df is not None:
        logging.info("Starting Step Success Rate calculation...")
        step_success_results = step_success(
            human_df.copy(),
            readable_agent_df.copy(),
            OPENAI_API_KEY,
            CONFIGURATION["openai_model"]
        )
        if step_success_results:
            # Convert list of dicts to DataFrame
            step_success_df = pd.DataFrame(step_success_results)

            # Add a prefix to avoid name clashes

            # Transpose
            df_transposed = step_success_df.T

            # Reset index to make Task No a column
            df_transposed = df_transposed.reset_index()

            # Rename the new column to 'Task No'
            df_transposed.rename(columns={'index': 'Task No'}, inplace=True)

            # Merge step success metrics into all_task_metrics based on 'Task No'
            all_task_metrics = all_task_metrics.merge(df_transposed, on='Task No', how='left')


        else:
            logging.info("Skipping Step Success Rate calculation.")

    else:
        logging.info("Skipping Step Success Rate")



    # --- Run Repetitive Rate for Agent ---
    if CONFIGURATION["run_repetitive_action_rate"] and agent_df is not None:
        repetitiveness_rate = repetitiveness(
            readable_agent_df.copy(),
            OPENAI_API_KEY,
            CONFIGURATION["openai_model"]
        )
        #all_task_metrics = merge_repetetive_results_into_output_df(all_task_metrics, repetitiveness_rate, task_col="Task No")
        # Check if the response is not empty or None
        if repetitiveness_rate:
            all_task_metrics = merge_repetetive_results_into_output_df(
                all_task_metrics,
                repetitiveness_rate,
                task_col="Task No"
            )
        else:
            logging.info("Repetitive Action Rate response is empty. Skipping merge.")

    else:
        logging.info("Skipping Repetitive Action Rate calculation as per configuration or missing data.")


    if CONFIGURATION["run_recovery_rate"]:
        if human_df is not None and agent_df is not None:
            logging.info("Starting Trajectory Re-alignment Score calculation...")

            trajectory_realignment_scores = recovery(
                human_df.copy(),
                readable_agent_df.copy(),
                OPENAI_API_KEY,
                CONFIGURATION["openai_model"],
                CONFIGURATION["semantic_similarity_threshold"],
                CONFIGURATION["max_agent_sequence_length"]
            )

            if trajectory_realignment_scores:
                # Convert per-task dictionary to DataFrame
                realignment_df = pd.DataFrame.from_dict(trajectory_realignment_scores, orient="index").reset_index()
                realignment_df.rename(columns={"index": "Task No"}, inplace=True)

                try:
                    all_task_metrics = merge_recovery_results_into_output_df(all_task_metrics,
                                                                             trajectory_realignment_scores)
                    logging.info(f"Detailed Recovery Score results saved")
                except Exception as e:
                    logging.error(f"Error saving detailed Trajectory Re-alignment Score Excel: {e}")

            else:
                logging.warning("No common tasks found for Trajectory Re-alignment Score calculation.")
        else:
            logging.info("Skipping Trajectory Re-alignment Score calculation due to missing human or agent data.")
    else:
        logging.info("Skipping Trajectory Re-alignment Score calculation as per configuration.")


    # --- Run Partial Success Rate ---
    if CONFIGURATION["run_partial_success_rate"] and agent_df is not None and keywords_df is not None:
        partial_success_results = partial_success(
            agent_df.copy(),
            keywords_df.copy(),
            OPENAI_API_KEY,
            CONFIGURATION["openai_model"]
        )
        if partial_success_results:
            try:
                all_task_metrics = merge_partial_success_into_output_df(all_task_metrics, partial_success_results, task_col="Task No")
                logging.info(f"Partial Success Rate results saved.")
            except Exception as e:
                logging.error(f"Error saving detailed Partial Success Ratet to df: {e}")
        else:
            logging.warning("Partial Success Rate calculation resulted in no data or an error.")
    else:
        logging.info("Skipping Partial Success Rate calculation as per configuration or missing data.")


    # --- Final Output ---
    summary_output_path = 'evaluation_metrics_per_task.xlsx'
    try:
        all_task_metrics.to_excel(summary_output_path, index=False)
        logging.info(f"All metrics per task saved to {summary_output_path}")
    except Exception as e:

        summary_output_path = 'evaluation_metrics_per_task.xlsx'
        logging.error(f"Error saving evaluation metrics Excel: {e}")
    logging.info("Evaluation script finished.")
