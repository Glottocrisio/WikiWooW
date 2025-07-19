function importModuleDescriptions() {
  // Get the form
  const form = FormApp.getActiveForm();
  
  // Open the spreadsheet by ID (replace with your spreadsheet ID)
  const spreadsheetId = "1fgW_SOdZwLbMIb2rV8EtBUlV0jXhm4hSsQ93NPGkVfs";
  const sheet = SpreadsheetApp.openById(spreadsheetId).getSheets()[0]; 
  
  const dataRange = sheet.getDataRange();
  const values = dataRange.getValues();
  
  // Column A + Column B: Module Name

  for (let i = 0; i < values.length; i++) {
    const moduleName = values[i][0]+ " " + values[i][1];
   
    
    if (moduleName) {
      // Create a new question for each module
      const item = form.addSectionHeaderItem()
        .setTitle(moduleName);
      
      // Add questions related to this module
      form.addMultipleChoiceItem()
        .setTitle(`Do you *suppose* there could be an *immediate* relationship between these entities?`)
        .setChoiceValues(['Yes', 'No']);
      
      form.addMultipleChoiceItem()
        .setTitle(`Do you already *know* the nature of the relationship?`)
        .setChoiceValues(['Yes', 'No']);

      form.addMultipleChoiceItem()
        .setTitle(`Would you *be interested* in knowing more about this relationship - if there were one?`)
        .setChoiceValues(['Yes', 'No']);
    }
  }
  
  Logger.log("Import completed");
}