from datetime import datetime, timedelta

class AgendaAssistant:
    def __init__(self):
        self.events = {}
        
    def add_event(self, title, date, duration, description=""):
        try:
            date = datetime.strptime(date, "%Y-%m-%d %H:%M")
            event = {
                "title": title,
                "date": date,
                "duration": duration,
                "description": description
            }
            if date not in self.events:
                self.events[date] = []
            self.events[date].append(event)
            return True
        except ValueError:
            return False
            
    def get_events_for_day(self, date):
        try:
            date = datetime.strptime(date, "%Y-%m-%d")
            day_events = []
            for event_date in self.events:
                if event_date.date() == date.date():
                    day_events.extend(self.events[event_date])
            return sorted(day_events, key=lambda x: x["date"])
        except ValueError:
            return []
