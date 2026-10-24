import pytest
from model import parse_sensor_json

def test_parse_sensor_json_valid():
    """Test that the parser correctly extracts variables and values from valid JSON."""
    sample_json = """[
        {"values": [{"time": 1709435097, "value": 23.3}], "variable": {"name": "Temperature"}},
        {"values": [{"time": 1709435097, "value": 426}], "variable": {"name": "Carbon dioxide"}}
    ]"""
    
    result = parse_sensor_json(sample_json)
    
    assert 'Temperature' in result
    assert result['Temperature'] == 23.3
    assert result['Carbon dioxide'] == 426

def test_parse_sensor_json_invalid():
    """Test that the parser gracefully handles broken or empty data."""
    broken_json = "this is not a json dictionary"
    result = parse_sensor_json(broken_json)
    
    assert result == {}
    assert isinstance(result, dict)