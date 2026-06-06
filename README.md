# soutez 2026

- je splneno nize popsane

### spusteni
- docker compose up --build
- aplikace bezi: 127.0.0.1:5000
- 
## reseni

### inicializace

- Získání sessionId a informací o dostupných vozidlech
- SessionId se používá pro všechny další požadavky

### parsovani mapy

- sazeni BMP souboru
- analyza pixelu: dle barvy
- detekce krizovatek
- rozdeleni mapy na bloky 10×10

### pozadavky
- ID, typ, startovni/cílova pozice, prioritu, cas startu

### hledani trasy

- A* algoritmus pro optimální cestu
- heuristika: Manhattan distance

### rizeni semeforu

- TrafficManager: spravuje stav na krizovatkach
- VehicleScheduler: plan prujezdu
- Nizsi priorita cislo = vyssi priorita

### simulace
- Tracking pozic vozidel
- Zaznamenávání stavu semaforu
- Generování protokolu pro server
