#expression to catch
mass_expression = {}
universal_float = r'\d+(\.)?(\d+)?(e[+-]\d+)?'

mass_expression[
    'total'] = r'reservoir[1-9]: Phase mass: \{ (' + universal_float + '), (' + universal_float + ') \} kg'
mass_expression[
    'trapped'] = r'reservoir[1-9]: Trapped phase mass \(metric 1\): \{ (' + universal_float + '), (' + universal_float + ') \} kg'
mass_expression[
    'immobile'] = r'reservoir[1-9]: Immobile phase mass \(metric 2\): \{ (' + universal_float + '), (' + universal_float + ') \} kg'
mass_expression[
    'mobile'] = r'reservoir[1-9]: Mobile phase mass \(metric 2\): \{ (' + universal_float + '), (' + universal_float + ') \} kg'
mass_expression[
    'dissolved'] = r'reservoir[1-9]: Dissolved component mass: \{ \{ (' + universal_float + '), (' + universal_float + ') \}, \{ (' + universal_float + '), (' + universal_float + ') \} \} kg'

#

newton_expression = {}

# newton_expression['dt'] = r', dt:(\s*[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?)'
newton_expression['ndt'] = r'New dt =(\s*[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?)'
newton_expression['adt'] = r'accepted dt =(\s*[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?)'
newton_expression['Iterations'] = r'Iterations: (\s*[1-9]\d*)'


