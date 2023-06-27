PvonXvorausgesetztY = [[0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9]];
x = [1; 1; 1; 0; 0; 0; 0; 0; 0;];
erg1 = PvonXvorausgesetztY * x;

PvonYvorausgesetztZ = [[0.9, 0.0333, 0.0333, 0.0333],
                         [0.0333, 0.9, 0.0333, 0.0333],
                         [0.0333, 0.0333, 0.9, 0.0333],
                         [0.0333, 0.0333, 0.0333, 0.9]];
z = [1; 0; 0; 0;];
erg2 = PvonYvorausgesetztZ * z;

erg3 = erg1 .* erg2;
% erg3 hat die wahrscheinlichkeiten für jedes yi drinnen, aber die
% normalisierung fehlt noch.

% zum normalisieren muss ich jede Zeile von erg3 durch die summe aller
% Zeilen von erg3 dividieren.
PvonYvorausgesetztXundZ = erg3 ./ sum(erg3, 1);

% zum Vergleich ohne Prior:
PvonYvorausgesetztX = erg1 ./ sum(erg1,1);

