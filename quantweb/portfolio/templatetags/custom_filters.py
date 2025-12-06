from django import template
import math

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Template filter to get dictionary item by key"""
    return dictionary.get(key, {})

@register.filter
def corr_color(value):
    """Convert correlation value to a color gradient"""
    try:
        value = float(value)
        # Ensure value is between -1 and 1
        value = max(-1, min(1, value))
        
        if value >= 0:
            # Positive correlation: blue to green gradient
            r = int(99 + (156 * (1 - value)))
            g = int(179 + (76 * value))
            b = int(246 - (156 * value))
        else:
            # Negative correlation: red to blue gradient
            r = int(239 + (16 * (1 + value)))
            g = int(68 + (111 * (1 + value)))
            b = int(68 + (178 * (1 + value)))
            
        return f'rgba({r}, {g}, {b}, 0.8)'
    except (ValueError, TypeError):
        return 'rgba(156, 163, 175, 0.8)'  # Default gray
