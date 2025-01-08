import re
from bs4 import BeautifulSoup

def extract_css(html_file):
    """
    Extracts inline <style> CSS and linked CSS files from an HTML file.
    """
    with open(html_file, 'r') as f:
        soup = BeautifulSoup(f, 'html.parser')
        
        # Extract inline CSS in <style> tags
        inline_css = ''.join([style.string or '' for style in soup.find_all('style')])
        return inline_css

def parse_css_rules(css_content):
    """
    Parses CSS rules and returns a list of dictionaries with selectors and declarations.
    """
    rules = []
    css_pattern = re.compile(r'([^{]+)\{([^}]+)\}')
    for match in css_pattern.finditer(css_content):
        selectors = [selector.strip() for selector in match.group(1).split(',')]
        declarations = {
            decl.split(':')[0].strip(): decl.split(':')[1].strip()
            for decl in match.group(2).split(';') if ':' in decl
        }
        rules.append({'selectors': selectors, 'declarations': declarations})
    return rules

def compute_specificity(selector):
    """
    Computes the specificity of a CSS selector as a tuple (inline, id, class, tag).
    """
    inline = 1 if 'style' in selector else 0
    id_count = selector.count('#')
    class_count = selector.count('.')
    tag_count = len(re.findall(r'^[a-zA-Z]+|(?<![.#])\b[a-zA-Z]+', selector))
    return (inline, id_count, class_count, tag_count)

def find_conflicts(rules_index, rules_query):
    """
    Finds conflicts between two sets of CSS rules.
    """
    conflicts = []
    for rule1 in rules_index:
        for selector1 in rule1['selectors']:
            specificity1 = compute_specificity(selector1)
            for rule2 in rules_query:
                for selector2 in rule2['selectors']:
                    specificity2 = compute_specificity(selector2)
                    if set(rule1['declarations']).intersection(rule2['declarations']):
                        for property in rule1['declarations']:
                            if property in rule2['declarations']:
                                winner = 'index.html' if specificity1 > specificity2 else 'query_results.html'
                                conflicts.append({
                                    'property': property,
                                    'index_rule': (selector1, rule1['declarations'][property]),
                                    'query_rule': (selector2, rule2['declarations'][property]),
                                    'winner': winner
                                })
    return conflicts

def explain_conflicts(conflicts):
    """
    Explains conflicts in plain English.
    """
    explanations = []
    for conflict in conflicts:
        explanation = (
            f"Conflict detected for property '{conflict['property']}':\n"
            f"- In `index.html`: selector `{conflict['index_rule'][0]}` with value `{conflict['index_rule'][1]}`.\n"
            f"- In `query_results.html`: selector `{conflict['query_rule'][0]}` with value `{conflict['query_rule'][1]}`.\n"
            f"Winner: {conflict['winner']} due to higher specificity or later declaration.\n"
        )
        explanations.append(explanation)
    return explanations

def main():
    # Extract CSS from both files
    index_css = extract_css('templates/index.html')
    query_css = extract_css('templates/query_results.html')
    
    # Parse CSS rules
    rules_index = parse_css_rules(index_css)
    rules_query = parse_css_rules(query_css)
    
    # Find conflicts
    conflicts = find_conflicts(rules_index, rules_query)
    
    # Explain conflicts
    explanations = explain_conflicts(conflicts)
    
    # Print explanations
    if explanations:
        print("\n".join(explanations))
    else:
        print("No conflicts detected.")

if __name__ == "__main__":
    main()