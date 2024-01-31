# RuleLoader
RuleLoader is a module that simplifies the process of defining and managing rules for labeling devices. The module serves as a parser for a simple configuration language described below.

## Basic usage

```python
> from src.experiments.ruleloader import RuleLoader
>
> loader = RuleLoader('/path/to/your/dp3_config/db_entities')
> label_rules = loader.parse_file('file_with_rules.conf')
```

RuleLoader takes care of reading and parsing a configuration file. In the example above, `label_rules` is a list of callable `Rule` and `RangeRule` objects, that take a single parameter, a record of the device's properties. `Rule` and `RangeRule` return either `None`, if the rule was not triggered, or a `MassFunction` object, expressing the resulting classification, and the belief assigned to it.

Here you can see the basic usage of label rules.

```python
> print(label_rules[0])  
Rule: {
    {Windows}:0.8; 
    {NAS, MacOS, AndroidPhone, Router, Server, ApplePhone, VoiceAssistant, Linux, Windows, SmartTV, Hub, Appliance}:0.2
} 
if any_of(['Windows', 'windows', 'win']) in r.operating_system_ua

> print(label_rules[0]({'operating_system_ua': ['Windows']}))
{
    {Windows}:0.8; 
    {NAS, MacOS, AndroidPhone, Router, Server, ApplePhone, VoiceAssistant, Linux, Windows, SmartTV, Hub, Appliance}:0.2
}

> print(label_rules[0]({'operating_system_ua': ['Linux']}))
None
```

For more information about `MassFunction` objects, see [PyDS library GitHub](https://github.com/reineking/pyds) or the bonus section at the end of [Demo Jupyter Notebook](https://gitlab.liberouter.org/xsedla1o/label-fusion/-/blob/0ec0953e289457080e820b9545a9f0b1cb885221/demo.ipynb).

## Configuration Language

The language was based on the one developed for [ConditionalLabels](https://redmine.liberouter.org/projects/adict/wiki/Conditional_Labeling), but the syntax has changed a lot. It doesn't include entities anymore, as they must be merged to enable cross-entity conditions. Whereas ConditionalLabelling assigned a specific value to a specific attribute after triggering its conditions, the `MassFunction` objects are to be merged with the results of other rules, placing more emphasis on probabilities. This additional nuance allows us to better classify devices, handle conflicts in data, as well as have a metric for how certain we are of a classification.

### Basics

The basic `Rule` definition pattern looks like this:
```
<Classifiacation>
	<Belief>: <Condition>
```

* `<Classification>` is a value from a class taxonomy, loaded from a configuration file. RuleLoader will load one from 
  `data/label_fusion_taxonomy.yaml` by default, but you can pass your own to RuleLoader as an optional argument `classes`.
  A dot notation is used to access deeper levels of a taxonomy - to select a class `Windows`, 
  which is subclass of `OperatingSystem`, use `OperatingSystem.Windows`.
* `<Belief>` is a float value in the interval <0, 1>, which will be assigned to `<Classification>` inside the returned `MassFunction`.
* The value of 1 - `<Belief>` will be assigned to all classes from `classes` enum.
`<Condition>` is a logical expression, if true, the `Rule` will be triggered.

Here is a concrete example:
```python
OperatingSystem.Windows
	0.8: any_of('Windows', 'windows', 'win') in .operating_system_ua
```

The `RangeRule` can be defined with a similar pattern:
```
<Classification>
	<BaseBelief> - <MaxBelief>: [
		<Condition1>
		<Condition2>
		...
	]
```

* `<BaseBelief>` and `<MaxBelief>` are again in interval <0, 1>, but the semantics are different. The `RangeRule` changes its belief for the given class based on how many conditions inside the square brackets are triggered. If no condition triggers, `None` is returned. On a single conditon triggering, the value of `<BaseBelief>` is assigned. In addition, the ratio of triggered conditions to all conditions, transfromed through an [easing function](https://easings.net/#easeOutQuart), multiplied with `<MaxBelief>-<BaseBelief>` is added on top. 
* Simply put, if all conditions are triggered, `<MaxBelief>` is assigned. If fewer conditions trigger, the value is interpolated between `<BaseBelief>` and `<MaxBelief>` based on their number.

Again, an example:
```python
OperatingSystem.Linux  
    0.4 - 0.7: [  
        '_ftp._tcp.local' in .dnssd_query  
        '_nfs._tcp.local' in .dnssd_query  
        '_webdavs._tcp.local' in .dnssd_query  
        '_webdav._tcp.local' in .dnssd_query  
    ]
```

The point of the nesting syntax is that you can add mulitple rules for one class without having to repeat its name. RuleLoader will still produce independent `Rule` and `RangeRule` instances. There's also support for writing comments, starting with `#`. The following example produces 1 `Rule` and 1 `RangeRule`. 
```python
# The server class is detected by the following rules
Device.Server  
    0.8: 'Server' in .tags_by_services
    # A comment at the rule level
    0.3 - 0.6: [
    	any_of('NETWORKSERVICE', 'MAIL') in .out_flow_tags
    	len(.shodan_monitor) > 0 # A comment following a rule
    ]
```

### Condition Syntax

Finally, a description of the conditions themselves. Conditions are called on with the evaluated record and return True or False. To access individual properties of the record use a dot notation: `.property_name`. 

For basic-type values (not structures), the comparison operators `<`, `>` and `==` are available.
```python
Device.Appliance
    0.5: .activity_flows < 10
    0.8: .hostname == 'Roomba'
```

When conditions apply to multiple classes equally, you can enumerate all such classes in a comma separated list.
Semantically, this will produce a classification with a set of enumerated classes as main item. 
```python
Device.Appliance, Device.NAS
    0.5: .activity_flows < 10
```

If your rule applies to all classes except a set few, you can use an exclamation mark (`!`) at the beginning of  
your list to select the complement of enumerated classes. The result is a classification with a set of
all except your enumerated classes as main item. 
```python
! Device.Appliance, Device.NAS
    0.5: .activity_flows > 10
```

For arrays, use the `in` operator.
```python
Device.Server  
    0.8: 'Server' in .tags_by_services
```

You can combine bool-returning expressions using logical operators `and` and `or`.
```python
OperatingSystem.Windows  
    0.2: .x > 0 or .y < 0.6
    0.5: .a == 'str' and (.b == 1 or .c == 1)
```

To extend the possibilites of conditions, a set of functions is a part of the language. 

The basic syntax is `<function name>(<arg0>, <arg1>, ...)`, where the number of args can go from 0 to N. Both constants and record properties can be passed as arguments. A single function can have multiple signatures with multiple implementations. Most currently available functions return a bool, but it is not  a rule. Here is an example using the `re` regex function.
```python
OperatingSystem.Windows  
    0.5: re('a+b?') in .y 
    0.5: re('a+b?', .a)
```

### More about the `in` operator
`in` requires an array or multivalue property on its right, and a constant or function on its left. The semantics are based on the python's `in` operator, but with additional support of pattern matching for structures. The array is iterated through, and individual elements are passed to the lefthand side of the operator for evaluation. The python code for this is quite simple:
```python
    # self.right(record) is the property access, and should return an array
    # self.left is the matching part
    return any(self.left(r) for r in self.right(record))
```

The `in` operator supports a dictionary notation for the `dict` and JSON datatype. The functionality will be explained on the following example.
```python
OperatingSystem.MacOS
    0.4 - 0.8: [
    	{'name': 'os', 'value': 'Mac OS X'} in .sdp_label
    	{'name': 'hostname', 'value': contains('MacBook')} in .sdp_label
    	{'service':'_ssh._tcp.local'} in .dnssd_service
    ]
```

`sdp_label` is defined as a `dict<name:string,value:string>`. The first condition is self-explanatory, A dictionary exactly as the one defined will match that condition. 

The second condition showcases the use of the `contains` function, as any compatible function can be applied inside the dictionary notation. It will match anything what has `'name'` equal to `'hostname'` and `'value'` which contains a substring `'MacBook'`.

The third condition is to show that not all keys must be specified if not required. `dnssd_service` is defined as `dict<port:int,service:string>`, but we don't really care about the port, so we can match just based on the service name.

### Functions

The point of functions is to extend the language using existing or custom functions. To enable type validation, the functions declare signatures. These signatures are the same as typical function signatures, except they include 2 sets of arguments. One is given inside the configuration file, through the constants and properties provided. The second is to be passed at runtime, as a value of the examined part of the record. A function can also be standalone, and handle its function without any additional runtime parameters. We will use `None` to denote this case.

Here is the description of currently available functions.

### `contains`

Provides a `'substring' in 'string'` functionality. Has 2 signatures:

*  `contains(substring: str) -> (string: str) -> (bool)`
	
	This signature expects a substring inside the configuration, and a string at runtime.
	
*  `contains(substring: str, string: str) -> (None) -> (bool)`

	This signature expects both the substring and string in config. The string should be a property for this to make sense.

```python
OperatingSystem.Windows
    0.5: contains('sub') in .y
    0.5: contains('sub', .a)
```

### `re`

Interface for the `re` module. The first argument will be passed to `re.compile`, the second or runtime argument will be compared with the resulting pattern with `re.match`. Signatures:

*  `re(pattern: str) -> (string: str) -> (bool)`
*  `re(pattern: str, string: str) -> (None) -> (bool)`

```python
OperatingSystem.Linux
    0.5: re('a+b?') in .y
    0.5: re('a+[^a]a+', .a)
```

### `contains_any_of`

Accepts any number of string arguments and returns whether any of them is a substring to the runtime argument. Has a single signature: 

* `contains_any_of(substring: str, ...) -> (str) -> (bool)`

```python
OperatingSystem.Linux
    0.8: contains_any_of('Linux', 'Debian', 'Ubuntu', 'CentOS', 'Fedora') in .operating_system_ua
```

### `len`

Interface for builtin `len` function. It is currently supported for strings and arrays of primitive types.

* `len(object: str) -> (None) -> (int)`
* `len(object: List[str]) -> (None) -> (int)`
* `len(object: List[int]) -> (None) -> (int)`
* `len(object: List[float]) -> (None) -> (int)`
```python
Device.Server  
    0.5: len(.shodan_monitor) > 0
```

### Extending Functions
The suite of implemented functions can be extended by implementing your own. New functions must be a subclass of abstract class `Function` and can be defined anywhere, where they will be initialized before the instantiation of RuleLoader. Here is a basic template:
```python
class FunctionName(Function):
    __slots__ = ()  # optionally add your slots here
    name = 'function_name_in_config'
 
    def __init__(self, args: List, function_implementation: Callable) -> None:
        super().__init__(args, function_implementation)
        # self.args contains the args, you can do any additional validation or setup here
 
    @classmethod
    def get_signature(cls) -> List[Tuple[Callable, Tuple[...], Optional[type], type]]:
        return [
            # example signature
            (FunctionName.implementation_method, (str,), str, int)
        ]
  
    def implementation_method(self, record):
        """ The function implementation goes here """
        pass
```

## Classification Taxonomy

The classes seen in this document are loaded at runtime from a YAML file.
It's default placement is `data/label_fusion_taxonomy.yaml`, but you may pass your own to the `RuleLoader` constructor,
using the `ClassificationTaxonomyManager` object and `load_classification_taxonomy()` from `classification_taxonomy.py`.

The YAML file can contain multiple taxonomy roots, each should contain nested dictionaries with the rest of the taxonomy.
Leaf classes should be indicated as dictionary entry with `null` value. You can see the format in the following example,
taken from the default config at the time of writing.

```yaml
Device:
  Workstation:
    Windows:
    Linux:
    MacOS:

  IoT:
    SmartTV:
    VoiceAssistant:
    Hub:
    Appliance:
```
